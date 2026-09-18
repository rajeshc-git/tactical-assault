# Stage 1: Build the game using Bun for lightning-fast speeds
FROM oven/bun:1 as builder

# Set the working directory inside the container
WORKDIR /app

# Copy package files (we copy these first to leverage Docker layer caching)
COPY package.json package-lock.json ./

# Install the project dependencies
RUN bun install

# Copy the rest of the project source code
COPY . .

# Build the optimized production bundle
RUN bun run build

# Stage 2: Serve the highly-optimized files using Nginx
FROM nginx:alpine

# Copy the static output from the builder stage over to Nginx's public folder
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port 80 to the outside world
EXPOSE 80

# Start Nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]
