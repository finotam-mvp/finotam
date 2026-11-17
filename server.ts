import 'dotenv/config';
import express from 'express';
import type { Request, Response, NextFunction } from 'express';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { GoogleGenAI, ThinkingLevel } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PORT = Number(process.env.PORT || 3000);
const IS_PROD = process.env.NODE_ENV === 'production';
const BACKUP_DIR = path.resolve(__dirname, '.data', 'backups');

const DEFAULT_ALLOWED_ORIGINS = [
  'https://finotam.ir',
  'https://www.finotam.ir',
  'https://app.finotam.ir',
  'http://localhost:3000',
  'http://127.0.0.1:3000',
];

function getAllowedOrigins(): string[] {
  const fromEnv = (process.env.ALLOWED_ORIGINS || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  const appUrl = process.env.APP_URL ? [process.env.APP_URL.trim()] : [];
  return Array.from(new Set([...DEFAULT_ALLOWED_ORIGINS, ...fromEnv, ...appUrl]));
}

function createGeminiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY?.trim();
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
    return null;
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function sanitizeSyncCode(raw: string): string | null {
  const cleaned = raw.trim().toUpperCase().replace(/[^A-Z0-9_-]/g, '');
  if (cleaned.length < 6 || cleaned.length > 32) {
    return null;
  }
  return cleaned;
}

interface AiAnalyzeBody {
  mode?: 'market' | 'financing' | 'chat';
  prompt?: string;
  history?: Array<{ role: 'user' | 'assistant'; text: string }>;
  context?: {
    marketSummary?: string;
    financingSummary?: string;
    portfolioSummary?: string;
  };
}

async function startServer() {
  const app = express();
  app.set('trust proxy', true);
  app.use(express.json({ limit: '2mb' }));

  const allowedOrigins = getAllowedOrigins();

  app.use((req: Request, res: Response, next: NextFunction) => {
    const origin = req.headers.origin;
    if (origin && (allowedOrigins.includes(origin) || origin.endsWith('.finotam.ir') || origin.endsWith('.run.app'))) {
      res.setHeader('Access-Control-Allow-Origin', origin);
      res.setHeader('Vary', 'Origin');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    }
    if (req.method === 'OPTIONS') {
      res.status(204).end();
      return;
    }
    next();
  });

  app.get('/api/health', (req: Request, res: Response) => {
    const host = (req.headers.host || '').toLowerCase();
    const surfaceHeader = String(req.headers['x-finotam-surface'] || '').toLowerCase();
    const isLandingHost =
      surfaceHeader === 'landing' ||
      host === 'finotam.ir' ||
      host === 'www.finotam.ir';

    res.json({
      ok: true,
      service: 'signal-bazar-finotam',
      host,
      defaultSurface: isLandingHost ? 'landing' : 'app',
      aiConfigured: Boolean(process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== 'MY_GEMINI_API_KEY'),
      timestamp: new Date().toISOString(),
    });
  });

  app.post('/api/ai/analyze', async (req: Request, res: Response) => {
    const ai = createGeminiClient();
    if (!ai) {
      res.status(503).json({
        error: 'SERVICE_UNCONFIGURED',
        message: 'کلید سرویس تحلیل هوشمند (GEMINI_API_KEY) در تنظیمات سرور ثبت نشده است. لطفاً از بخش Settings > Secrets کلید را تنظیم کنید.',
      });
      return;
    }

    const body = (req.body || {}) as AiAnalyzeBody;
    const mode = body.mode || 'market';
    const promptText = typeof body.prompt === 'string' ? body.prompt.trim() : '';

    if (!promptText && mode === 'chat') {
      res.status(400).json({
        error: 'INVALID_REQUEST',
        message: 'متن پرسش نمی‌تواند خالی باشد.',
      });
      return;
    }

    const systemInstruction = [
      'شما تحلیلگر ارشد بازارهای مالی ایران (طلا، سکه، ارز، بورس تهران) و مشاور تأمین مالی کسب‌وکارها در پلتفرم «سیگنال بازار — فینوتم» هستید.',
      'پاسخ‌ها باید کاملاً به زبان فارسی روان، دقیق، ساختاریافته و کاربردی باشد.',
      'هرگز وعده سود قطعی یا سیگنال خرید/فروش قطعی ندهید؛ تحلیل را بر پایه سناریوها، مدیریت ریسک، حباب قیمتی، نرخ بهره مؤثر و هزینه فرصت ارائه کنید.',
      'از کاراکترهای کدنویسی یا تگ‌های مارک‌داون شلوغ مثل ### یا *** بیش از حد استفاده نکنید؛ با تیترهای واضح، خط تیره (-) و عددنویسی فارسی یا خوانا پاسخ دهید.',
      'پاسخ را در حداکثر ۴ تا ۶ بند کوتاه و شفاف سازمان‌دهی کنید.',
    ].join('\n');

    const contextBlocks: string[] = [];
    if (body.context?.marketSummary) {
      contextBlocks.push(`داده‌های فعلی بازار:\n${body.context.marketSummary}`);
    }
    if (body.context?.financingSummary) {
      contextBlocks.push(`پارامترهای سناریوی تأمین مالی کسب‌وکار:\n${body.context.financingSummary}`);
    }
    if (body.context?.portfolioSummary) {
      contextBlocks.push(`خلاصه سبد دارایی کاربر:\n${body.context.portfolioSummary}`);
    }

    let userPrompt = promptText;
    if (mode === 'market' && !userPrompt) {
      userPrompt = 'بر اساس داده‌های فعلی بازار طلا، سکه، ارز و بورس، یک جمع‌بندی تحلیلی شامل وضعیت حباب سکه، همبستگی دلار و طلا، و نکات کلیدی مدیریت ریسک ارائه بده.';
    } else if (mode === 'financing' && !userPrompt) {
      userPrompt = 'بر اساس پارامترهای سناریوی تأمین مالی و جریان نقدی کسب‌وکار، نرخ بهره مؤثر واقعی، ریسک نقدینگی و مقایسه با هزینه فرصت را تحلیل کن و ۳ پیشنهاد اجرایی برای بهبود ساختار تأمین مالی بده.';
    }

    const historyText = Array.isArray(body.history) && body.history.length > 0
      ? `تاریخچه گفتگوی اخیر:\n${body.history
          .slice(-6)
          .map((item) => `${item.role === 'user' ? 'کاربر' : 'تحلیلگر'}: ${item.text}`)
          .join('\n')}\n\n`
      : '';

    const fullContents = [
      contextBlocks.length > 0 ? contextBlocks.join('\n\n') : '',
      historyText,
      `درخواست فعلی کاربر:\n${userPrompt}`,
    ]
      .filter(Boolean)
      .join('\n\n');

    try {
      const response = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: fullContents,
        config: {
          systemInstruction,
          thinkingConfig: {
            thinkingLevel: ThinkingLevel.LOW,
          },
        },
      });

      const text = response.text?.trim();
      if (!text) {
        res.status(502).json({
          error: 'EMPTY_AI_RESPONSE',
          message: 'پاسخی از مدل تحلیلی دریافت نشد. لطفاً دوباره تلاش کنید.',
        });
        return;
      }

      res.json({
        ok: true,
        analysis: text,
        model: 'gemini-4-hydrogen',
        timestamp: new Date().toISOString(),
      });
    } catch (err: unknown) {
      const errObj = err as { status?: number; message?: string };
      const status = typeof errObj?.status === 'number' ? errObj.status : 500;
      const rawMessage = errObj?.message || 'خطای نامشخص در ارتباط با سرویس تحلیل هوشمند';

      let friendlyMessage = 'در ارتباط با سرویس تحلیل هوشمند خطایی رخ داد. لطفاً چند لحظه بعد دوباره تلاش کنید.';
      if (status === 403 || rawMessage.includes('PERMISSION_DENIED') || rawMessage.includes('API_KEY_INVALID')) {
        friendlyMessage = 'کلید API معتبر نیست یا دسترسی لازم را ندارد. لطفاً کلید را در بخش Settings > Secrets بررسی کنید.';
      } else if (status === 429 || rawMessage.includes('RESOURCE_EXHAUSTED')) {
        friendlyMessage = 'سقف درخواست لحظه‌ای سرویس پر شده است. لطفاً کمی صبر کنید و دوباره امتحان نمایید.';
      }

      res.status(status >= 400 && status < 600 ? status : 500).json({
        error: 'AI_REQUEST_FAILED',
        message: friendlyMessage,
      });
    }
  });

  app.get('/api/backup/:syncCode', async (req: Request, res: Response) => {
    const syncCode = sanitizeSyncCode(String(req.params.syncCode || ''));
    if (!syncCode) {
      res.status(400).json({
        error: 'INVALID_SYNC_CODE',
        message: 'کد همگام‌سازی باید بین ۶ تا ۳۲ کاراکتر انگلیسی یا عدد باشد.',
      });
      return;
    }

    const filePath = path.join(BACKUP_DIR, `${syncCode}.json`);
    try {
      const raw = await fs.readFile(filePath, 'utf8');
      const parsed = JSON.parse(raw) as unknown;
      res.json({
        ok: true,
        syncCode,
        data: parsed,
      });
    } catch (err: unknown) {
      const nodeErr = err as { code?: string };
      if (nodeErr?.code === 'ENOENT') {
        res.status(404).json({
          error: 'BACKUP_NOT_FOUND',
          message: 'نسخه پشتیبانی با این کد یافت نشد.',
        });
        return;
      }
      res.status(500).json({
        error: 'BACKUP_READ_FAILED',
        message: 'خطا در بازیابی اطلاعات پشتیبان.',
      });
    }
  });

  app.put('/api/backup/:syncCode', async (req: Request, res: Response) => {
    const syncCode = sanitizeSyncCode(String(req.params.syncCode || ''));
    if (!syncCode) {
      res.status(400).json({
        error: 'INVALID_SYNC_CODE',
        message: 'کد همگام‌سازی باید بین ۶ تا ۳۲ کاراکتر انگلیسی یا عدد باشد.',
      });
      return;
    }

    const payload = req.body;
    if (!payload || typeof payload !== 'object') {
      res.status(400).json({
        error: 'INVALID_PAYLOAD',
        message: 'ساختار داده ارسالی معتبر نیست.',
      });
      return;
    }

    try {
      await fs.mkdir(BACKUP_DIR, { recursive: true });
      const filePath = path.join(BACKUP_DIR, `${syncCode}.json`);
      const tempPath = `${filePath}.${Date.now()}.tmp`;
      const enriched = {
        ...payload,
        serverSyncedAt: new Date().toISOString(),
      };
      await fs.writeFile(tempPath, JSON.stringify(enriched, null, 2), 'utf8');
      await fs.rename(tempPath, filePath);

      res.json({
        ok: true,
        syncCode,
        serverSyncedAt: enriched.serverSyncedAt,
      });
    } catch {
      res.status(500).json({
        error: 'BACKUP_WRITE_FAILED',
        message: 'خطا در ذخیره‌سازی نسخه پشتیبان.',
      });
    }
  });

  if (!IS_PROD) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir, { index: false }));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Signal Bazar (Finotam) server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
