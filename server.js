import express from "express";
import path from "path";
import dotenv from "dotenv";
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { createProxyMiddleware } from 'http-proxy-middleware';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const port = process.env.PORT || 3000;

app.use('/api', createProxyMiddleware({ 
    target: 'http://127.0.0.1:8000', 
    changeOrigin: true,
    pathRewrite: {
        '^/api': '',
    },
    logLevel: 'debug',
}));

app.use(express.static(path.join(__dirname, '/')));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(port, () => {
  console.log(`Node.js app listening on port ${port}`);
});