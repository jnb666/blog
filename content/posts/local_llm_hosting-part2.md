+++
date = '2025-08-27T19:15:27+01:00'
title = 'Local LLM models: Part2 - a basic cmd line app in go'
draft = true
+++
For the first post in this series which covers setting up the environment see [part 1](/posts/local_llm_hosting-part1).

The `llama-server` binary provides an OpenAI compatible Completions API using a REST + JSON interface at 
`/v1/completions` URL. In this post we test this out by writing a few command line programs in go.

<!--more-->

To save a lot of boilerplate we'll use the [go-openai module](https://pkg.go.dev/github.com/sashabaranov/go-openai)
which handles building and parsing the necessary messages. 
(There is also a newer go SDK from openai, but it's much more painful to use).
```go
import openai "github.com/sashabaranov/go-openai"
```

The first thing we need is a client object which will handle making the HTTP requests. 
The BaseURL needs to point to the endpoint on our local server, we can leave the auth token blank as it's not needed.

```go
	config := openai.DefaultConfig("")
	config.BaseURL = "http://localhost:8080/v1"
	client := openai.NewClientWithConfig(config)
```

```go
package main

import (
	"context"
	"fmt"

	openai "github.com/sashabaranov/go-openai"
)

func main() {
	config := openai.DefaultConfig("")
	config.BaseURL = "http://localhost:8080/v1"
	client := openai.NewClientWithConfig(config)

	chatCompletion, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{
			{Role: "user", Content: "Why is the sky blue?"},
		},
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println(chatCompletion.Choices[0].Message.Content)
}
```

We'll create a new go module for this with below directory structure:
```
gpt-go/
├── cmd
│   ├── simple
│   │   └── simple.go
│   └── test
├── go.mod
└── go.sum
```

