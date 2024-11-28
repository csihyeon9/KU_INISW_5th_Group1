## FastAPI for Fin￦i$E Services
Developed by Soohyun Kwon, Keumhwan Kim, Seohyun Park, Junhyeok Lee, Changhee Cho, and Sihyun Cha from the 5th cohort of Korea University INISW Academy.

### Usage
Obtain an HTTPS certificate using Let's Encrypt's Certbot in an Ubuntu 22.04 LTS environment.

1. Install Docker
    ```
    sudo apt update
    sudo apt install docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    ```
1. Build the Docker Image 
    ```
    docker build -t finwise-backend .
    ```
1. Run the Container
    ```
    docker run -d --name finwise-backend -p 8000:8000 -v /etc/letsencrypt:/etc/letsencrypt finwise-backend
    ```

### License
MIT License.