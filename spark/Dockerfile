FROM openjdk:11-jre-slim

# Spark 설치
RUN apt-get update && apt-get install -y curl python3 python3-pip
RUN curl -O https://downloads.apache.org/spark/spark-3.4.4/spark-3.4.4-bin-hadoop3.tgz && \
    tar -xvzf spark-3.4.4-bin-hadoop3.tgz && \
    mv spark-3.4.4-bin-hadoop3 /opt/spark && \
    rm spark-3.4.4-bin-hadoop3.tgz
ENV SPARK_HOME=/opt/spark
ENV PATH="$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH"

# PyTorch 및 PySpark 설치
RUN pip3 install torch torchvision pyspark

# 스크립트 복사
COPY start-spark.sh /opt/start-spark.sh
RUN chmod +x /opt/start-spark.sh

CMD ["/opt/start-spark.sh"]