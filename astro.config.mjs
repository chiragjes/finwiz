// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';


// https://astro.build/config
export default defineConfig({
	integrations: [
		starlight({
			title: 'FinLearn',
			social: {
				github: 'https://github.com/withastro/starlight',
			},
			sidebar: [
				{
					label: 'Notebooks',
					autogenerate: {directory: 'notebooks'},
				},

				{
					label: 'Reference',
					autogenerate: { directory: 'reference' },
				},
			],
		}),
	],
});
