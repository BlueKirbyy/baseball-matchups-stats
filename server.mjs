import http from 'node:http';
import { readFile } from 'node:fs/promises';

const port = Number(process.env.PORT || 8000);
const upstream = 'https://api.prizepicks.com/projections?league_id=2&per_page=250&single_stat=true&game_mode=pickem';

const server = http.createServer(async (request, response) => {
  if (request.url === '/api/projections') {
    try {
      const upstreamResponse = await fetch(upstream, {
        headers: {
          accept: 'application/json',
          'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138 Safari/537.36',
          referer: 'https://app.prizepicks.com/',
        },
      });
      const body = await upstreamResponse.text();
      response.writeHead(upstreamResponse.status, {
        'content-type': upstreamResponse.headers.get('content-type') || 'application/json',
        'cache-control': 'no-store',
      });
      response.end(body);
    } catch (error) {
      response.writeHead(502, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ error: 'Unable to reach the PrizePicks projections feed.' }));
    }
    return;
  }

  if (request.url === '/' || request.url === '/index.html') {
    const page = await readFile(new URL('./index.html', import.meta.url));
    response.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
    response.end(page);
    return;
  }

  response.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
  response.end('Not found');
});

server.listen(port, () => console.log(`MLB Lines is running at http://localhost:${port}`));
