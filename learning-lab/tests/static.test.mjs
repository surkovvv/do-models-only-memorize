import test from "node:test";
import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

import { LESSONS } from "../js/lessons.js";

const LAB_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const REPO_ROOT = path.resolve(LAB_ROOT, "..");

test("lesson ids are unique and referenced production sources exist", async () => {
  assert.equal(new Set(LESSONS.map((lesson) => lesson.id)).size, LESSONS.length);
  for (const lesson of LESSONS) {
    const sourcePath = path.resolve(LAB_ROOT, lesson.sourceHref);
    const sourceStat = await stat(sourcePath);
    assert.equal(sourceStat.isFile(), true, `${lesson.id}: missing ${sourcePath}`);
    assert.equal(sourcePath.startsWith(REPO_ROOT), true);
  }
});

test("static shell has landmark, skip link, and module entrypoint", async () => {
  const html = await readFile(path.join(LAB_ROOT, "index.html"), "utf8");

  assert.match(html, /lang="ru"/);
  assert.match(html, /class="skip-link"/);
  assert.match(html, /id="lab-app"/);
  assert.match(html, /type="module" src="\.\/js\/app\.js"/);
});

test("stylesheet includes narrow layout and reduced-motion fallback", async () => {
  const css = await readFile(path.join(LAB_ROOT, "styles.css"), "utf8");

  assert.match(css, /@media \(max-width: 680px\)/);
  assert.match(css, /prefers-reduced-motion: reduce/);
  assert.match(css, /:focus-visible/);
});

