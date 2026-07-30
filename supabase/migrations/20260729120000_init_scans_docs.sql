-- Abjadi: scans, OCR results, documentation
-- Run in Supabase SQL Editor (or via supabase db push)

create extension if not exists "pgcrypto";

-- ---------------------------------------------------------------------------
-- profiles (optional until Auth is enabled)
-- ---------------------------------------------------------------------------
create table if not exists public.profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  display_name text,
  created_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- scans
-- ---------------------------------------------------------------------------
create table if not exists public.scans (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references public.profiles (id) on delete set null,
  public_id text not null unique,
  source_image_path text,
  overlay_image_path text,
  status text not null default 'pending'
    check (status in ('pending', 'analyzed', 'failed', 'documented')),
  avg_confidence integer check (avg_confidence is null or (avg_confidence >= 0 and avg_confidence <= 100)),
  api_device text,
  error text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists scans_user_id_idx on public.scans (user_id);
create index if not exists scans_created_at_idx on public.scans (created_at desc);

-- ---------------------------------------------------------------------------
-- ocr_results (1:1 with scans)
-- ---------------------------------------------------------------------------
create table if not exists public.ocr_results (
  id uuid primary key default gen_random_uuid(),
  scan_id uuid not null unique references public.scans (id) on delete cascade,
  raw_text text not null default '',
  n_lines integer not null default 0,
  n_glyphs integer not null default 0,
  lines jsonb not null default '[]'::jsonb,
  glyphs jsonb not null default '[]'::jsonb,
  mode text not null default 'paper',
  device text,
  raw_payload jsonb,
  created_at timestamptz not null default now()
);

create index if not exists ocr_results_scan_id_idx on public.ocr_results (scan_id);

-- ---------------------------------------------------------------------------
-- documentations
-- ---------------------------------------------------------------------------
create table if not exists public.documentations (
  id uuid primary key default gen_random_uuid(),
  scan_id uuid not null unique references public.scans (id) on delete cascade,
  user_id uuid references public.profiles (id) on delete set null,
  public_id text not null unique,
  title text not null default '',
  script_type text not null default '',
  language text not null default '',
  region text not null default '',
  country text not null default '',
  description text not null default '',
  condition text not null default '',
  image_source text not null default '',
  ocr_text_edited text not null default '',
  notes text not null default '',
  confidence integer check (confidence is null or (confidence >= 0 and confidence <= 100)),
  status text not null default 'submitted'
    check (status in ('draft', 'submitted', 'under_review', 'published')),
  submitted_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists documentations_user_id_idx on public.documentations (user_id);
create index if not exists documentations_submitted_at_idx on public.documentations (submitted_at desc);

-- ---------------------------------------------------------------------------
-- documentation_images
-- ---------------------------------------------------------------------------
create table if not exists public.documentation_images (
  id uuid primary key default gen_random_uuid(),
  documentation_id uuid not null references public.documentations (id) on delete cascade,
  storage_path text not null,
  is_primary boolean not null default false,
  sort_order integer not null default 0,
  created_at timestamptz not null default now()
);

create index if not exists documentation_images_doc_id_idx
  on public.documentation_images (documentation_id);

-- ---------------------------------------------------------------------------
-- updated_at helper
-- ---------------------------------------------------------------------------
create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists scans_set_updated_at on public.scans;
create trigger scans_set_updated_at
  before update on public.scans
  for each row execute function public.set_updated_at();

drop trigger if exists documentations_set_updated_at on public.documentations;
create trigger documentations_set_updated_at
  before update on public.documentations
  for each row execute function public.set_updated_at();

-- ---------------------------------------------------------------------------
-- Storage buckets (private)
-- ---------------------------------------------------------------------------
insert into storage.buckets (id, name, public)
values
  ('scan-images', 'scan-images', false),
  ('documentation-images', 'documentation-images', false)
on conflict (id) do nothing;

-- ---------------------------------------------------------------------------
-- RLS: enabled; service role bypasses. Client policies ready for Auth later.
-- ---------------------------------------------------------------------------
alter table public.profiles enable row level security;
alter table public.scans enable row level security;
alter table public.ocr_results enable row level security;
alter table public.documentations enable row level security;
alter table public.documentation_images enable row level security;

-- Own profile
drop policy if exists "profiles_select_own" on public.profiles;
create policy "profiles_select_own" on public.profiles
  for select using (auth.uid() = id);

drop policy if exists "profiles_update_own" on public.profiles;
create policy "profiles_update_own" on public.profiles
  for update using (auth.uid() = id);

-- Own scans
drop policy if exists "scans_select_own" on public.scans;
create policy "scans_select_own" on public.scans
  for select using (auth.uid() = user_id);

drop policy if exists "scans_insert_own" on public.scans;
create policy "scans_insert_own" on public.scans
  for insert with check (auth.uid() = user_id);

drop policy if exists "scans_update_own" on public.scans;
create policy "scans_update_own" on public.scans
  for update using (auth.uid() = user_id);

-- OCR via scan ownership
drop policy if exists "ocr_select_own" on public.ocr_results;
create policy "ocr_select_own" on public.ocr_results
  for select using (
    exists (
      select 1 from public.scans s
      where s.id = ocr_results.scan_id and s.user_id = auth.uid()
    )
  );

drop policy if exists "docs_select_own" on public.documentations;
create policy "docs_select_own" on public.documentations
  for select using (auth.uid() = user_id);

drop policy if exists "docs_insert_own" on public.documentations;
create policy "docs_insert_own" on public.documentations
  for insert with check (auth.uid() = user_id);

drop policy if exists "docs_update_own" on public.documentations;
create policy "docs_update_own" on public.documentations
  for update using (auth.uid() = user_id);

drop policy if exists "doc_images_select_own" on public.documentation_images;
create policy "doc_images_select_own" on public.documentation_images
  for select using (
    exists (
      select 1 from public.documentations d
      where d.id = documentation_images.documentation_id and d.user_id = auth.uid()
    )
  );

-- Storage: authenticated users can manage their own folder prefix {uid}/...
drop policy if exists "scan_images_own" on storage.objects;
create policy "scan_images_own" on storage.objects
  for all using (
    bucket_id = 'scan-images'
    and auth.role() = 'authenticated'
    and (storage.foldername(name))[1] = auth.uid()::text
  )
  with check (
    bucket_id = 'scan-images'
    and auth.role() = 'authenticated'
    and (storage.foldername(name))[1] = auth.uid()::text
  );

drop policy if exists "documentation_images_own" on storage.objects;
create policy "documentation_images_own" on storage.objects
  for all using (
    bucket_id = 'documentation-images'
    and auth.role() = 'authenticated'
    and (storage.foldername(name))[1] = auth.uid()::text
  )
  with check (
    bucket_id = 'documentation-images'
    and auth.role() = 'authenticated'
    and (storage.foldername(name))[1] = auth.uid()::text
  );
