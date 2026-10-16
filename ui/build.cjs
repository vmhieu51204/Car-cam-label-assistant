const esbuild = require('esbuild');
const sass = require('sass');
const path = require('path');
const fs = require('fs');
const root = __dirname;
const output = path.join(root, 'dist');
fs.mkdirSync(output, { recursive: true });
esbuild.build({
  absWorkingDir: root, entryPoints: ['workspace.ts'], bundle: true, minify: true,
  platform: 'browser', format: 'iife', target: ['chrome99'], outfile: path.join(output, 'workspace.js'),
  plugins: [{ name: 'cvat-scss', setup(build) {
    build.onLoad({ filter: /\.scss$/ }, args => ({
      contents: sass.compile(args.path, { style: 'compressed', silenceDeprecations: ['legacy-js-api', 'import', 'global-builtin', 'color-functions'] }).css,
      loader: 'css', resolveDir: path.dirname(args.path)
    }));
  }}],
}).then(() => {
  fs.copyFileSync(path.join(root, 'index.html'), path.join(output, 'index.html'));
  fs.copyFileSync(path.join(root, 'vendor/cvat-source.json'), path.join(output, 'cvat-source.json'));
}).catch(error => { console.error(error); process.exitCode = 1; });
