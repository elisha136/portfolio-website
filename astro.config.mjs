import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';

// https://astro.build/config
export default defineConfig({
  // Replace with your actual domain when you have one
  site: 'https://elisha-antwi.dev',
  
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
