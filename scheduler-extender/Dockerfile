FROM docker.io/library/golang:1.20 AS builder
WORKDIR /go/src/app
COPY . .
RUN go mod init extender && go build -o extender .
CMD ["./extender"]
