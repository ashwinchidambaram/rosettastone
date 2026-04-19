.PHONY: css css-watch

css:
	npx tailwindcss -i src/rosettastone/server/static/css/tailwind-input.css \
		-o src/rosettastone/server/static/css/tailwind.css --minify

css-watch:
	npx tailwindcss -i src/rosettastone/server/static/css/tailwind-input.css \
		-o src/rosettastone/server/static/css/tailwind.css --watch
