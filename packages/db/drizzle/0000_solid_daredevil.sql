CREATE TYPE "public"."documentation_status" AS ENUM('draft', 'submitted', 'under_review', 'published');--> statement-breakpoint
CREATE TYPE "public"."scan_status" AS ENUM('pending', 'analyzed', 'failed', 'documented');--> statement-breakpoint
CREATE TABLE "documentation_images" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"documentation_id" uuid NOT NULL,
	"storage_path" text NOT NULL,
	"is_primary" boolean DEFAULT false NOT NULL,
	"sort_order" integer DEFAULT 0 NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "documentations" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"scan_id" uuid NOT NULL,
	"user_id" uuid,
	"public_id" text NOT NULL,
	"title" text DEFAULT '' NOT NULL,
	"script_type" text DEFAULT '' NOT NULL,
	"language" text DEFAULT '' NOT NULL,
	"region" text DEFAULT '' NOT NULL,
	"country" text DEFAULT '' NOT NULL,
	"description" text DEFAULT '' NOT NULL,
	"condition" text DEFAULT '' NOT NULL,
	"image_source" text DEFAULT '' NOT NULL,
	"ocr_text_edited" text DEFAULT '' NOT NULL,
	"notes" text DEFAULT '' NOT NULL,
	"confidence" integer,
	"status" "documentation_status" DEFAULT 'submitted' NOT NULL,
	"submitted_at" timestamp with time zone DEFAULT now() NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "documentations_scan_id_unique" UNIQUE("scan_id"),
	CONSTRAINT "documentations_public_id_unique" UNIQUE("public_id")
);
--> statement-breakpoint
CREATE TABLE "ocr_results" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"scan_id" uuid NOT NULL,
	"raw_text" text DEFAULT '' NOT NULL,
	"n_lines" integer DEFAULT 0 NOT NULL,
	"n_glyphs" integer DEFAULT 0 NOT NULL,
	"lines" jsonb DEFAULT '[]'::jsonb NOT NULL,
	"glyphs" jsonb DEFAULT '[]'::jsonb NOT NULL,
	"mode" text DEFAULT 'paper' NOT NULL,
	"device" text,
	"raw_payload" jsonb,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "ocr_results_scan_id_unique" UNIQUE("scan_id")
);
--> statement-breakpoint
CREATE TABLE "profiles" (
	"id" uuid PRIMARY KEY NOT NULL,
	"display_name" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "scans" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid,
	"public_id" text NOT NULL,
	"source_image_path" text,
	"overlay_image_path" text,
	"status" "scan_status" DEFAULT 'pending' NOT NULL,
	"avg_confidence" integer,
	"api_device" text,
	"error" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "scans_public_id_unique" UNIQUE("public_id")
);
--> statement-breakpoint
ALTER TABLE "documentation_images" ADD CONSTRAINT "documentation_images_documentation_id_documentations_id_fk" FOREIGN KEY ("documentation_id") REFERENCES "public"."documentations"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "documentations" ADD CONSTRAINT "documentations_scan_id_scans_id_fk" FOREIGN KEY ("scan_id") REFERENCES "public"."scans"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "documentations" ADD CONSTRAINT "documentations_user_id_profiles_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."profiles"("id") ON DELETE set null ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "ocr_results" ADD CONSTRAINT "ocr_results_scan_id_scans_id_fk" FOREIGN KEY ("scan_id") REFERENCES "public"."scans"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "scans" ADD CONSTRAINT "scans_user_id_profiles_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."profiles"("id") ON DELETE set null ON UPDATE no action;