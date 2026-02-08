# Ubuntu base image
FROM ubuntu:22.04

# Avoid interactive apt prompts
ENV DEBIAN_FRONTEND=noninteractive

# ---- System deps: Python, Java, tools ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        default-jre-headless \
        wget \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# ---- Weka + Bounce (dependency) ----
# Weka stable JAR from Maven Central
ARG WEKA_VERSION=3.8.6
ARG WEKA_M2_GROUP_PATH=nz/ac/waikato/cms/weka/weka-stable

# Bounce (Weka third-party) JAR from Maven Central
ARG BOUNCE_VERSION=0.18
ARG BOUNCE_M2_GROUP_PATH=nz/ac/waikato/cms/weka/thirdparty/bounce

RUN mkdir -p /opt/weka && \
    wget -O /opt/weka/weka.jar \
      "https://repo1.maven.org/maven2/${WEKA_M2_GROUP_PATH}/${WEKA_VERSION}/weka-stable-${WEKA_VERSION}.jar" && \
    wget -O /opt/weka/bounce.jar \
      "https://repo1.maven.org/maven2/${BOUNCE_M2_GROUP_PATH}/${BOUNCE_VERSION}/bounce-${BOUNCE_VERSION}.jar" && \
    chmod 644 /opt/weka/*.jar

# NOTE: WEKA_JAR is now a *classpath*, not a single file path
ENV WEKA_JAR=/opt/weka/weka.jar:/opt/weka/bounce.jar

# Optional helper to run J48 directly
RUN echo '#!/usr/bin/env bash\n' \
    'java -cp "$WEKA_JAR" weka.classifiers.trees.J48 "$@"\n' \
    > /usr/local/bin/weka-j48 && \
    chmod +x /usr/local/bin/weka-j48

# ---- App setup ----
WORKDIR /opt/ga-hls

# Copy the repo into the container
COPY . /opt/ga-hls

# Install ga-hls and its Python deps
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -e .

# Default outputs dir (can be overridden via env)
ENV GA_HLS_OUTPUT_DIR=/opt/ga-hls/outputs
RUN mkdir -p "${GA_HLS_OUTPUT_DIR}"

# ---- Entrypoint ----
ENTRYPOINT ["ga-hls"]
CMD ["--help"]
