# Domain Shift Testbed

### Tested on...
```
pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
```
### Requirements
```
tqdm
tensorboard
xformers
timm
```
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