import { relations } from "drizzle-orm";
import {
  boolean,
  integer,
  jsonb,
  pgEnum,
  pgTable,
  text,
  timestamp,
  uuid,
} from "drizzle-orm/pg-core";

export const scanStatusEnum = pgEnum("scan_status", [
  "pending",
  "analyzed",
  "failed",
  "documented",
]);

export const documentationStatusEnum = pgEnum("documentation_status", [
  "draft",
  "submitted",
  "under_review",
  "published",
]);

/** Optional until Supabase Auth is wired; FK to auth.users added in extras SQL */
export const profiles = pgTable("profiles", {
  id: uuid("id").primaryKey(),
  displayName: text("display_name"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export const scans = pgTable("scans", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: uuid("user_id").references(() => profiles.id, { onDelete: "set null" }),
  publicId: text("public_id").notNull().unique(),
  sourceImagePath: text("source_image_path"),
  overlayImagePath: text("overlay_image_path"),
  status: scanStatusEnum("status").notNull().default("pending"),
  avgConfidence: integer("avg_confidence"),
  apiDevice: text("api_device"),
  error: text("error"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
});

export const ocrResults = pgTable("ocr_results", {
  id: uuid("id").primaryKey().defaultRandom(),
  scanId: uuid("scan_id")
    .notNull()
    .unique()
    .references(() => scans.id, { onDelete: "cascade" }),
  rawText: text("raw_text").notNull().default(""),
  nLines: integer("n_lines").notNull().default(0),
  nGlyphs: integer("n_glyphs").notNull().default(0),
  lines: jsonb("lines").notNull().default([]),
  glyphs: jsonb("glyphs").notNull().default([]),
  mode: text("mode").notNull().default("paper"),
  device: text("device"),
  rawPayload: jsonb("raw_payload"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export const documentations = pgTable("documentations", {
  id: uuid("id").primaryKey().defaultRandom(),
  scanId: uuid("scan_id")
    .notNull()
    .unique()
    .references(() => scans.id, { onDelete: "cascade" }),
  userId: uuid("user_id").references(() => profiles.id, { onDelete: "set null" }),
  publicId: text("public_id").notNull().unique(),
  title: text("title").notNull().default(""),
  scriptType: text("script_type").notNull().default(""),
  language: text("language").notNull().default(""),
  region: text("region").notNull().default(""),
  country: text("country").notNull().default(""),
  description: text("description").notNull().default(""),
  condition: text("condition").notNull().default(""),
  imageSource: text("image_source").notNull().default(""),
  ocrTextEdited: text("ocr_text_edited").notNull().default(""),
  notes: text("notes").notNull().default(""),
  confidence: integer("confidence"),
  status: documentationStatusEnum("status").notNull().default("submitted"),
  submittedAt: timestamp("submitted_at", { withTimezone: true }).notNull().defaultNow(),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
});

export const documentationImages = pgTable("documentation_images", {
  id: uuid("id").primaryKey().defaultRandom(),
  documentationId: uuid("documentation_id")
    .notNull()
    .references(() => documentations.id, { onDelete: "cascade" }),
  storagePath: text("storage_path").notNull(),
  isPrimary: boolean("is_primary").notNull().default(false),
  sortOrder: integer("sort_order").notNull().default(0),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

/**
 * Store-binary compatibility policy for the mobile app (`GET /mobile/config`).
 * Exactly one row should be `is_active = true` (enforced in extras.sql).
 */
export const mobileVersionPolicies = pgTable("mobile_version_policies", {
  id: uuid("id").primaryKey().defaultRandom(),
  isActive: boolean("is_active").notNull().default(false),
  iosMinimumSupportedVersion: text("ios_minimum_supported_version").notNull(),
  iosLatestVersion: text("ios_latest_version").notNull(),
  iosStoreUrl: text("ios_store_url").notNull().default(""),
  androidMinimumSupportedVersion: text("android_minimum_supported_version").notNull(),
  androidLatestVersion: text("android_latest_version").notNull(),
  androidStoreUrl: text("android_store_url").notNull().default(""),
  updateMessageAr: text("update_message_ar").notNull(),
  updateMessageEn: text("update_message_en").notNull(),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
});

export const scansRelations = relations(scans, ({ one, many }) => ({
  user: one(profiles, { fields: [scans.userId], references: [profiles.id] }),
  ocrResult: one(ocrResults, { fields: [scans.id], references: [ocrResults.scanId] }),
  documentation: one(documentations, {
    fields: [scans.id],
    references: [documentations.scanId],
  }),
}));

export const ocrResultsRelations = relations(ocrResults, ({ one }) => ({
  scan: one(scans, { fields: [ocrResults.scanId], references: [scans.id] }),
}));

export const documentationsRelations = relations(documentations, ({ one, many }) => ({
  scan: one(scans, { fields: [documentations.scanId], references: [scans.id] }),
  user: one(profiles, { fields: [documentations.userId], references: [profiles.id] }),
  images: many(documentationImages),
}));

export const documentationImagesRelations = relations(documentationImages, ({ one }) => ({
  documentation: one(documentations, {
    fields: [documentationImages.documentationId],
    references: [documentations.id],
  }),
}));
