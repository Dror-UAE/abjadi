CREATE TABLE "mobile_version_policies" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"is_active" boolean DEFAULT false NOT NULL,
	"ios_minimum_supported_version" text NOT NULL,
	"ios_latest_version" text NOT NULL,
	"ios_store_url" text DEFAULT '' NOT NULL,
	"android_minimum_supported_version" text NOT NULL,
	"android_latest_version" text NOT NULL,
	"android_store_url" text DEFAULT '' NOT NULL,
	"update_message_ar" text NOT NULL,
	"update_message_en" text NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL
);
