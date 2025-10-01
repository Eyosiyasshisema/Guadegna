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
from langgraph.graph import START, MessagesState, StateGraph
from typing import cast 


load_dotenv()

# --- ENVIRONMENT VARIABLE CHECKS ---
if "LANGSMITH_API_KEY" not in os.environ or not os.getenv("LANGSMITH_API_KEY"):
    raise EnvironmentError("LANGSMITH_API_KEY environment variable is missing. Deployment aborted.")

if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = "default"

if "GOOGLE_API_KEY" not in os.environ or not os.getenv("GOOGLE_API_KEY"):
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

# --------------------------------------------------------------------------------
# --- GLOBAL INITIALIZATION: ROBUST CHECKPOINTER CONFIGURATION ---
redis_client = None 
memory = None
is_redis_compatible = False
temp_redis_client = None

try:
    if REDIS_URL:
        # 1. ATTEMPT CONNECTION
        temp_redis_client = redis.from_url(
            REDIS_URL,
            db=0,
            decode_responses=False, 
            health_check_interval=15, 
            retry_on_timeout=True,
            socket_timeout=10, 
            max_connections=50 
        )
        temp_redis_client.ping()

        # 2. CHECK COMPATIBILITY 
        try:
            temp_redis_client.module_list() 
            is_redis_compatible = True
        except redis.exceptions.ResponseError as module_error:
            if "MODULE" in str(module_error):
                print("INFO: Redis connection established, but 'MODULE' command is unsupported. Falling back to MemorySaver.")
                temp_redis_client.close()
                temp_redis_client = None
            else:
                raise module_error
        
except Exception as e: 
    print(f"INFO: Redis connection/initialization failed: {str(e)}. Falling back to MemorySaver.")
    temp_redis_client = None
    is_redis_compatible = False

# 3. SET GLOBAL CHECKPOINTER BASED ON RESULT
if is_redis_compatible and temp_redis_client:
    from langgraph.checkpoint.redis import RedisSaver
    memory = RedisSaver(redis_client=temp_redis_client) 
    redis_client = temp_redis_client
    print("INFO: Successfully configured RedisSaver.")
else:
    # Use MemorySaver in all other cases
    from langgraph.checkpoint.memory import MemorySaver 
    memory = MemorySaver()
    redis_client = None
    if not REDIS_URL:
        print("INFO: REDIS_URL was not set. Defaulting to in-memory chat history (MemorySaver).")

# --- END OF GLOBAL INITIALIZATION ---
# --------------------------------------------------------------------------------


def get_disposable_redis_client():
    """
    Creates a new, short-lived Redis client instance for critical reads.
    Returns None if global redis_client is None or REDIS_URL is not set.
    """
    if not REDIS_URL or not redis_client: 
        return None
        
    return redis.from_url(
        REDIS_URL, 
        db=0, 
        decode_responses=False,
        socket_timeout=5, 
        socket_connect_timeout=5,
        max_connections=1 
    )

def add_to_token_blacklist(token: str, expiration: datetime):
    if redis_client:
        try:
            now = datetime.now(timezone.utc)
            time_to_expire = int((expiration - now).total_seconds())
            if time_to_expire > 0:
                redis_client.set(f"blacklist:{token}".encode('utf-8'), b"revoked", ex=time_to_expire) 
                return True
        except redis.exceptions.ConnectionError as e:
            return False
    return False

def is_token_blacklisted(token: str):
    r = get_disposable_redis_client() 
    if not r:
        return False
        
    try:
        r.ping() 
        return r.get(f"blacklist:{token}".encode('utf-8')) is not None
    except redis.exceptions.ConnectionError:
        return False 
    except Exception:
        return False

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

def get_current_user(token: str = Depends(oauth2_scheme)):
    if is_token_blacklisted(token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has been revoked")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")
    except jwt.JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")
    return user_id


# --- DELETED OLD save_initial_chat_context FUNCTION (replaced by SQL logic) ---


def get_chat_history(user_id: str):
    """
    Retrieves chat history from LangGraph and configuration from the SQL database.
    """
    history = []
    ideal_friend = None
    buddy_name = None
    
    # 1. Get initial context (ideal_friend, buddy_name) from SQL Database (Primary Source)
    try:
        with Session(engine) as session:
            statement = select(User).where(User.id == uuid.UUID(user_id))
            user = session.exec(statement).first()
            if user:
                # Use the new DB fields
                ideal_friend = user.ideal_friend_config
                buddy_name = user.buddy_name
    except Exception:
        # If DB read fails, assume no config
        pass
            
    # 2. If configuration is available, try to retrieve chat history from LangGraph checkpointer
    if ideal_friend and buddy_name: 
        config = {"configurable": {"thread_id": user_id}}
        try:
            # memory is guaranteed to be either RedisSaver (compatible) or MemorySaver
            raw_state = memory.get(config) 
            messages_channel = raw_state.get('channel_values') if raw_state else None
            
            if messages_channel and messages_channel.get('messages'):
                messages = messages_channel['messages']
                for msg in messages:
                    if isinstance(msg, SystemMessage):
                        continue 

                    sender_type = 'human' if isinstance(msg, HumanMessage) else 'ai'
                    
                    history.append({
                        "sender": sender_type,
                        "content": msg.content
                    })
        except Exception:
            pass

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
    # --- NEW FIELDS FOR PERSISTENT CONFIGURATION (FIX) ---
    ideal_friend_config: str | None = Field(default=None)
    buddy_name: str | None = Field(default=None)
    # ----------------------------------------------------

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    has_buddy: bool = False 

class ChatRequest(BaseModel):
    message: str
    # FIX: Make personality fields optional for subsequent messages (Fixes 422 error)
    ideal_friend: str | None = None 
    buddy_name: str | None = None

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

@app.post("/api/signup", response_model=LoginResponse)
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
            hashed_password=hashed_password,
            # New users have no config yet
            ideal_friend_config=None,
            buddy_name=None
        )
        session.add(new_user)
        session.commit()
        session.refresh(new_user)

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(new_user.id)}, expires_delta=access_token_expires
        )
        # New users always return has_buddy=False
        return {"access_token": access_token, "token_type": "bearer", "has_buddy": False}
    
@app.post("/api/login", response_model=LoginResponse)
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

    # FIX: Check config directly from the User object (DB). (Fixes incorrect redirection)
    has_buddy = bool(user.ideal_friend_config and user.buddy_name) 

    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "has_buddy": has_buddy
    }

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
    """
    Fetches the conversation history and the ideal friend context.
    """
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

# Model is kept globally as it is stateless and efficient to reuse
from langchain_google_genai import ChatGoogleGenerativeAI

def create_compiled_app(user_id: str, current_ideal_friend: str, memory_saver):
    """
    Creates and compiles the LangGraph app with the appropriate memory saver.
    FIX: The prompt is dynamically created using the user's current_ideal_friend.
    """
    prompt_template = create_prompt_template(current_ideal_friend)
    
    workflow = StateGraph(state_schema=MessagesState)

    def call_model(state: MessagesState):
        prompt = prompt_template.invoke(state)
        response = model.invoke(prompt)
        return {"messages": [response]}

    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)
    compiled_app = workflow.compile(checkpointer=memory_saver)
    return compiled_app

    
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, user_id: str = Depends(get_current_user)):
    user_message = request.message
    
    stored_context = get_chat_history(user_id) 
    stored_ideal_friend = stored_context.get('ideal_friend')
    stored_buddy_name = stored_context.get('buddy_name')
    
    current_ideal_friend = stored_ideal_friend
    current_buddy_name = stored_buddy_name

    # --- 1. HANDLE INITIAL CONFIGURATION (if context is missing) ---
    if not stored_ideal_friend:
        # Enforce the requirement only on the first setup call
        if not request.ideal_friend or not request.buddy_name:
             raise HTTPException(status_code=400, detail="Buddy personality (ideal_friend) and name are required for first chat call.")

        current_ideal_friend = request.ideal_friend
        current_buddy_name = request.buddy_name
        
        # FIX: Save context permanently to SQL DB
        try:
            with Session(engine) as session:
                user_id_uuid = uuid.UUID(user_id)
                user = session.get(User, user_id_uuid)
                if user:
                    user.ideal_friend_config = current_ideal_friend
                    user.buddy_name = current_buddy_name
                    session.add(user)
                    session.commit()
        except Exception as e:
            print(f"WARNING: Failed to save buddy configuration to database: {e}")
    
    if not current_ideal_friend:
        # Should not happen after the check above, but as a final guard
        raise HTTPException(status_code=400, detail="Buddy personality (ideal_friend) is required.")
        
    # --- 2. COMPILE AND INVOKE CHAT (with error handling) ---
    try:
        # FIX: Use the dynamic creator to avoid stale data (stale data fix)
        compiled_app = create_compiled_app(user_id, current_ideal_friend, memory) 

        config = {"configurable": {"thread_id": user_id}}

        input_messages = {"messages": [HumanMessage(content=user_message)]}
        output = compiled_app.invoke(input_messages, config)
        
        model_response = output["messages"][-1]
        
        return {"response": model_response.content}
        
    except Exception as e:
        # Graceful failure on chat execution
        print(f"ERROR during compiled_app.invoke or state saving: {e}")
        return {"response": "Oops! I ran into an issue while trying to think of a response or save our chat. Please try again in a moment."}