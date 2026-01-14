-- InsightForge AI - User Authentication Table
-- Simple fake authentication for demo purposes (no real security)
-- Run this in Supabase SQL editor

CREATE TABLE IF NOT EXISTS public.if_users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,  -- Plain text for demo (no security concerns)
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP
);

-- Create index on username for faster lookups
CREATE INDEX IF NOT EXISTS idx_if_users_username ON public.if_users(username);

-- Optional: Add a comment
COMMENT ON TABLE public.if_users IS 'Simple user authentication table for demo purposes. Passwords stored in plain text.';
