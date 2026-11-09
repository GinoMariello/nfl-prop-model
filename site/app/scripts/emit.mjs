// Turn Vite's single-file build into the page fragment the artifact publish expects (the publish wraps
// it in its own document): title + font links + inlined style + root + inlined script, written to
// site/index.html beside data.json.
import { readFileSync, writeFileSync } from "node:fs";

const html = readFileSync(new URL("../dist/index.html", import.meta.url), "utf8");
const pick = re => [...html.matchAll(re)].map(m => m[0]).join("\n");
const out = [
  '<meta charset="utf-8">',
  pick(/<title>[\s\S]*?<\/title>/g),
  pick(/<meta name="claude:[^>]*>/g),
  pick(/<link [^>]*>/g),
  pick(/<style[^>]*>[\s\S]*?<\/style>/g),
  '<div id="root"></div>',
  pick(/<script[^>]*>[\s\S]*?<\/script>/g),
].join("\n");
if (!out.includes("<script") || !out.includes("<style")) throw new Error("build output is missing its script or style");
writeFileSync(new URL("../../index.html", import.meta.url), out + "\n");
console.log(`site/index.html written (${(out.length / 1024).toFixed(0)} KB)`);
