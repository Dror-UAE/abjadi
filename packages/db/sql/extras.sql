-- Run after Drizzle migrations: triggers, auth FK, RLS, storage buckets

create extension if not exists "pgcrypto";

-- indexes
create index if not exists scans_user_id_idx on public.scans (user_id);
create index if not exists scans_created_at_idx on public.scans (created_at desc);
create index if not exists ocr_results_scan_id_idx on public.ocr_results (scan_id);
create index if not exists documentations_user_id_idx on public.documentations (user_id);
create index if not exists documentations_submitted_at_idx on public.documentations (submitted_at desc);
create index if not exists documentation_images_doc_id_idx
  on public.documentation_images (documentation_id);

-- Exactly one active mobile version policy
create unique index if not exists mobile_version_policies_one_active_idx
  on public.mobile_version_policies (is_active)
  where is_active = true;

-- profiles → auth.users
do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'profiles_id_fkey'
  ) then
    alter table public.profiles
      add constraint profiles_id_fkey
      foreign key (id) references auth.users (id) on delete cascade;
  end if;
end $$;

-- updated_at triggers
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

drop trigger if exists mobile_version_policies_set_updated_at on public.mobile_version_policies;
create trigger mobile_version_policies_set_updated_at
  before update on public.mobile_version_policies
  for each row execute function public.set_updated_at();

-- Default / seed: one active mobile version policy (1.0.0)
insert into public.mobile_version_policies (
  is_active,
  ios_minimum_supported_version,
  ios_latest_version,
  ios_store_url,
  android_minimum_supported_version,
  android_latest_version,
  android_store_url,
  update_message_ar,
  update_message_en
)
select
  true,
  '1.0.0',
  '1.0.0',
  '',
  '1.0.0',
  '1.0.0',
  '',
  'يتوفر إصدار جديد من أبجدي. يرجى تحديث التطبيق للمتابعة.',
  'A new version of Abjadi is available. Please update to continue.'
where not exists (
  select 1 from public.mobile_version_policies where is_active = true
);

-- storage buckets
insert into storage.buckets (id, name, public)
values
  ('scan-images', 'scan-images', false),
  ('documentation-images', 'documentation-images', false)
on conflict (id) do nothing;

-- RLS
alter table public.profiles enable row level security;
alter table public.scans enable row level security;
alter table public.ocr_results enable row level security;
alter table public.documentations enable row level security;
alter table public.documentation_images enable row level security;
-- Service-role only (API). No anon/authenticated policies.
alter table public.mobile_version_policies enable row level security;

drop policy if exists "profiles_select_own" on public.profiles;
create policy "profiles_select_own" on public.profiles
  for select using (auth.uid() = id);

drop policy if exists "profiles_update_own" on public.profiles;
create policy "profiles_update_own" on public.profiles
  for update using (auth.uid() = id);

drop policy if exists "scans_select_own" on public.scans;
create policy "scans_select_own" on public.scans
  for select using (auth.uid() = user_id);

drop policy if exists "scans_insert_own" on public.scans;
create policy "scans_insert_own" on public.scans
  for insert with check (auth.uid() = user_id);

drop policy if exists "scans_update_own" on public.scans;
create policy "scans_update_own" on public.scans
  for update using (auth.uid() = user_id);

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
