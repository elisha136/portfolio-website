/**
 * Content Collections Configuration
 * Defines schemas for project content
 */
import { defineCollection, z } from 'astro:content';

// Projects collection schema
const projectsCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string(),
    category: z.string(),
    status: z.enum(['completed', 'ongoing', 'planned']),
    duration: z.string(),
    role: z.string(),
    institution: z.string(),
    image: z.string(),
    technologies: z.array(z.string()),
    github: z.string().optional(),
    featured: z.boolean().default(false),
    publishedDate: z.date().optional(),
  }),
});

export const collections = {
  projects: projectsCollection,
};

