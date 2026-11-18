// Demo convenience: forward /scan-api/* from the Expo dev server to the local scan server,
// so the phone only needs to reach one address (the one it already loads the app from).
const { getDefaultConfig } = require('expo/metro-config');
const http = require('node:http');

const config = getDefaultConfig(__dirname);
const SCAN_SERVER = process.env.SCAN_SERVER ?? 'http://127.0.0.1:8000';
const PREFIX = '/scan-api';

const previous = config.server?.enhanceMiddleware;
config.server = {
  ...config.server,
  enhanceMiddleware: (middleware, server) => {
    const base = previous ? previous(middleware, server) : middleware;
    return (req, res, next) => {
      if (!req.url?.startsWith(`${PREFIX}/`)) return base(req, res, next);
      const target = new URL(req.url.slice(PREFIX.length), SCAN_SERVER);
      const upstream = http.request(
        target,
        { method: req.method, headers: { ...req.headers, host: target.host } },
        (upRes) => {
          res.writeHead(upRes.statusCode ?? 502, upRes.headers);
          upRes.pipe(res);
        },
      );
      upstream.on('error', () => {
        res.writeHead(502, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ error: `Scan server not running at ${SCAN_SERVER} (run ./run-demo.sh)` }));
      });
      req.pipe(upstream);
    };
  },
};

module.exports = config;
