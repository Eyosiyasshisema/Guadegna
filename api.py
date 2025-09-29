import os
import uuid
from jose import jwt
import redis
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from passlib.context import CryptContext
from sqlmodel import SQLModel, Field, create_engine, Session, select
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()

if "LANGSMITH_API_KEY" not in os.environ:
    if not os.getenv("LANGSMITH_API_KEY"):
        raise EnvironmentError("LANGSMITH_API_KEY environment variable is missing. Deployment aborted.")

if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = "default"

if "GOOGLE_API_KEY" not in os.environ:
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY environment variable is missing. Deployment aborted.")
        
if "SECRET_KEY" not in os.environ:
    raise EnvironmentError("SECRET_KEY environment variable is missing. Deployment aborted.")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise EnvironmentError("DATABASE_URL environment variable is missing. Database connection aborted.")

app = FastAPI()

origins = [
    "https://guadegnabuddy.netlify.app", 
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            
    allow_credentials=True,           
    allow_methods=["*"],              
    allow_headers=["*"],              
)

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") 

engine = create_engine(DATABASE_URL, echo=False)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

REDIS_URL = os.getenv("REDIS_URL")

redis_client = None 
try:
    if not REDIS_URL:
        raise redis.exceptions.ConnectionError("REDIS_URL not set in environment.")
    redis_client = redis.from_url(
        REDIS_URL,
        db=0,
        decode_responses=True,
        health_check_interval=15, 
        retry_on_timeout=True ,
        socket_timeout=10, 
        max_connections=50  
    )
    redis_client.ping()
    memory = RedisSaver(redis_client=redis_client) 
except redis.exceptions.ConnectionError as e:
    print(f"Could not connect to Redis: {e}")
    print("Falling back to in-memory storage. Chat history will not persist.")
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()

def get_disposable_redis_client():
    """Creates a new, short-lived Redis client instance."""
    if not REDIS_URL:
        raise redis.exceptions.ConnectionError("REDIS_URL not set for disposable client.")
        
    return redis.from_url(
        REDIS_URL, 
        db=0, 
        decode_responses=True, 
        socket_timeout=10,
        max_connections=1 
    )


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def add_to_token_blacklist(token: str, expiration: datetime):
    if redis_client:
        try:
            now = datetime.now(timezone.utc)
            time_to_expire = int((expiration - now).total_seconds())
            if time_to_expire > 0:
                redis_client.set(f"blacklist:{token}", "revoked", ex=time_to_expire)
                return True
        except redis.exceptions.ConnectionError as e:
            print(f"Warning: Failed to blacklist token using global client: {e}")
            return False
    return False

def is_token_blacklisted(token: str):
    """
    Checks if a token is blacklisted using a disposable client. 
    This is the CRITICAL change to fix ConnectionError in auth flow.
    """
    if not REDIS_URL:
        return False 
        
    try:
        r = get_disposable_redis_client() 
        return r.get(f"blacklist:{token}") is not None
    except redis.exceptions.ConnectionError as e:
        print(f"CRITICAL REDIS FAILURE in is_token_blacklisted: {e}")
        return False 
    except Exception as e:
        print(f"Unexpected error in is_token_blacklisted: {e}")
        return False


def get_current_user(token: str = Depends(oauth2_scheme)):
    if is_token_blacklisted(token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has been revoked")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")
    except jwt.JWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")
    return user_id


def save_initial_chat_context(user_id: str, ideal_friend: str, buddy_name: str):
    """Saves the user's defined ideal friend characteristics and name to Redis."""
    if redis_client:
        try:
            context_data = {
                "ideal_friend": ideal_friend,
                "buddy_name": buddy_name
            }
            redis_client.hmset(f"context:{user_id}", context_data) 
        except redis.exceptions.ConnectionError as e:
            print(f"Warning: Failed to save chat context using global client: {e}")
    else:
        print("Warning: Cannot save chat context. Redis client not available.")

def get_chat_history(user_id: str):
    """Retrieves chat history and ideal friend characteristics from Redis/LangGraph."""
    history = []
    ideal_friend = None
    buddy_name = None
    context_data = {}

    if redis_client:
        try:
            context_data = redis_client.hgetall(f"context:{user_id}")
            
            ideal_friend = context_data.get(b"ideal_friend")
            buddy_name = context_data.get(b"buddy_name")
            
            if isinstance(ideal_friend, bytes):
                ideal_friend = ideal_friend.decode('utf-8')
            if isinstance(buddy_name, bytes):
                buddy_name = buddy_name.decode('utf-8')
            
            if ideal_friend and buddy_name: 
                buddy_name = buddy_name or "Buddy"
                config = {"configurable": {"thread_id": user_id}}
                try:
                    raw_state = memory.get(config) 
                    messages_channel = raw_state.get('channel_values') if raw_state else None
                    
                    if messages_channel and messages_channel.get('messages'):
                        messages = messages_channel['messages']
                        for msg in messages:
                            if isinstance(msg, SystemMessage):
                                continue 

                            sender_type = 'human'
                            if isinstance(msg, AIMessage):
                                sender_type = 'ai'

                            history.append({
                                "sender": sender_type,
                                "content": msg.content
                            })
                except Exception as e:
                    print(f"Warning: Failed to retrieve LangGraph history for user {user_id}: {e}")
        except redis.exceptions.ConnectionError as e:
            print(f"Warning: Failed to retrieve chat context using global client: {e}")

    return {
        "ideal_friend": ideal_friend,
        "buddy_name": buddy_name,
        "history": history
    }


class User(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    username: str = Field(unique=True, index=True)
    email: str = Field(unique=True, index=True)
    hashed_password: str

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatRequest(BaseModel):
    message: str
    ideal_friend: str
    buddy_name: str 

class ChatHistoryResponse(BaseModel):
    ideal_friend: str | None = None
    buddy_name: str | None = None
    history: list[dict]

@app.on_event("startup")
def on_startup():
    create_db_and_tables()


@app.get("/")
def read_root():
    return {"message": "Welcome to the Buddy API! Access /docs for documentation."}

@app.get("/users")
def list_users(current_user: str = Depends(get_current_user)):
    return {"message": "List of users (Placeholder)", "current_user_id": current_user}

@app.post("/api/signup", response_model=Token)
def signup(user_data: UserCreate):
    with Session(engine) as session:
        statement = select(User).where(User.email == user_data.email)
        existing_user = session.exec(statement).first()
        if existing_user:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

        hashed_password = get_password_hash(user_data.password)
        new_user = User(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password
        )
        session.add(new_user)
        session.commit()
        session.refresh(new_user)

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(new_user.id)}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/login", response_model=Token)
def login(user_data: UserCreate):
    with Session(engine) as session:
        statement = select(User).where(User.email == user_data.email)
        user = session.exec(statement).first()
        if not user or not verify_password(user_data.password, user.hashed_password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user.id)}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/logout")
def logout(request: Request, user_id: str = Depends(get_current_user)):
    """Revokes the current access token by adding it to a Redis blacklist."""
    
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing or invalid Authorization header")

    token = auth_header.split(" ")[1]

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        expiration = datetime.fromtimestamp(payload.get("exp"), tz=timezone.utc)
    except jwt.JWTError:
        return {"message": "Logout successful (Token was invalid, but session cleared)"}

    if add_to_token_blacklist(token, expiration):
        return {"message": "Logout successful (Token revoked)"}
    else:
        return {"message": "Logout successful (Server-side revocation skipped)"}
        
@app.get("/api/history", response_model=ChatHistoryResponse)
async def get_history_endpoint(user_id: str = Depends(get_current_user)):
    """Fetches the conversation history and the ideal friend context from the checkpoint."""
    history_data = get_chat_history(user_id)
    
    if history_data['ideal_friend'] and history_data['buddy_name']:
        return {
            "ideal_friend": history_data['ideal_friend'],
            "buddy_name": history_data['buddy_name'], 
            "history": history_data['history']
        }
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, 
        detail="Buddy personality not configured for this user."
    )
    
def create_prompt_template(ideal_friend: str):
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are an imaginary friend with the following personality and characteristics: {ideal_friend} You should act as their friend and never mention being an AI or a bot. Your responses should reflect the characteristics provided by the user.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, user_id: str = Depends(get_current_user)):
    user_message = request.message
    stored_context = get_chat_history(user_id) 
    stored_ideal_friend = stored_context.get('ideal_friend')
    stored_buddy_name = stored_context.get('buddy_name')
    
    current_ideal_friend = stored_ideal_friend if stored_ideal_friend else request.ideal_friend
    current_buddy_name = stored_buddy_name if stored_buddy_name else request.buddy_name

    if not stored_ideal_friend:
        if not request.ideal_friend or not request.buddy_name:
             raise HTTPException(status_code=400, detail="Buddy personality (ideal_friend) and name are required for first chat call.")

        save_initial_chat_context(user_id, current_ideal_friend, current_buddy_name)
    
    if not current_ideal_friend:
        raise HTTPException(status_code=400, detail="Buddy personality (ideal_friend) is required.")

    prompt_template = create_prompt_template(current_ideal_friend)
    
    workflow = StateGraph(state_schema=MessagesState)

    def call_model(state: MessagesState):
        prompt = prompt_template.invoke(state)
        response = model.invoke(prompt)
        return {"messages": [response]}

    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    compiled_app = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": user_id}}

    input_messages = {"messages": [HumanMessage(content=user_message)]}
    output = compiled_app.invoke(input_messages, config)
    
    model_response = output["messages"][-1]
    
    return {"response": model_response.content}