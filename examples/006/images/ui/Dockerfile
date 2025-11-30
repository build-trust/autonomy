FROM cgr.dev/chainguard/node:latest-dev AS builder
WORKDIR /app
COPY --chown=node:node package*.json ./
RUN npm ci
COPY --chown=node:node . .
RUN npm run build

FROM cgr.dev/chainguard/node:latest
WORKDIR /app
COPY --chown=node:node --from=builder /app/.next/standalone ./
COPY --chown=node:node --from=builder /app/.next/static ./.next/static
ENV HOSTNAME="0.0.0.0"
ENV PORT=8000
CMD ["server.js"]
