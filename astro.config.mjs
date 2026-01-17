import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';

// https://astro.build/config
export default defineConfig({
  site: 'https://www.elishadiburo.com',
  
  integrations: [
    tailwind(),
    mdx(),
    sitemap()
  ],
  
  markdown: {
    shikiConfig: {
      // Choose from: github-dark, github-light, nord, one-dark-pro, etc.
      theme: 'github-dark',
      wrap: true
    }
  }
});
