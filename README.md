# Domain Shift Testbed

```yaml
# docker-compose.yaml example
version: '2.3'
services:
  dev:
    image: 
    working_dir: /workspace
    build:
      context: .
    shm_size:
    stdin_open: true
    tty: true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 8
              capabilities: [ gpu ]
    volumes:
      - .:/workspace
      - type: bind
        source: /disk0/username
        target: /workspace/disk0
```