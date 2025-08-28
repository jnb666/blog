+++
date = '2025-08-28T13:52:27+01:00'
title = 'Local LLM models: Part2 - a basic cmd line app in go'
+++
For the first post in this series which covers setting up the environment see [part 1](/posts/local_llm_hosting-part1).

The `llama-server` binary provides an OpenAI compatible Completions API using a REST + JSON interface at 
`/v1/completions` URL. In this post we test this out by writing some simple command line programs in go.

<!--more-->

### Sending a chat completion request

All the code below is available at the [github.com/jnb666/gpt-go](https://github.com/jnb666/gpt-go) repo.
To save a lot of boilerplate we'll use the 
[github.com/sashabaranov/go-openai](https://pkg.go.dev/github.com/sashabaranov/go-openai) module
which handles building and parsing the necessary messages. 
(There is also a newer go SDK from openai, but it's much more painful to use).

This is the minimal code we need to connect to the local server, send a request with the chat so far 
and get the response back from the model.

[hello.go](https://github.com/jnb666/gpt-go/tree/main/cmd/hello/hello.go)
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

	req := openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{
			{Role: "user", Content: "Hello"},
		},
	}
	resp, err := client.CreateChatCompletion(context.Background(), req)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("== analysis ==\n%s\n== final ==\n%s\n",
		resp.Choices[0].Message.ReasoningContent, resp.Choices[0].Message.Content)
}
```

```
$ go run cmd/hello/hello.go
== analysis ==
User says "Hello". Probably just greeting. We can respond with greeting. Then perhaps ask how we can help.
== final ==
Hello! 👋 How can I help you today?
```

### Using the streaming API

That's simple enough but `CreateChatCompletion` waits for the entire response to be generated before returning. 
To show the tokens as they are produced we need to use the streaming API.

This function will make a streaming request, send updates with new text as each new token is generated
by the model and accumulate and return the final result:
```go
func createChatCompletionStream(ctx context.Context, client *openai.Client, request openai.ChatCompletionRequest,
	callback func(openai.ChatCompletionStreamChoiceDelta)) (choice openai.ChatCompletionChoice, err error) {

	stream, err := client.CreateChatCompletionStream(ctx, request)
	if err != nil {
		return choice, err
	}
	defer stream.Close()
	for {
		resp, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return choice, nil
		}
		if err != nil {
			return choice, err
		}
		if len(resp.Choices) == 0 {
			continue
		}
		delta := resp.Choices[0].Delta
		callback(delta)
		choice.Message.Content += delta.Content
		choice.Message.ReasoningContent += delta.ReasoningContent
		if resp.Choices[0].FinishReason != "" {
			choice.FinishReason = resp.Choices[0].FinishReason
		}
	}
}
```

An simple callback function we can plug into this to print the updates to stdout is:
```go
func printOutput() func(openai.ChatCompletionStreamChoiceDelta) {
	channel := ""
	return func(delta openai.ChatCompletionStreamChoiceDelta) {
		if delta.ReasoningContent != "" {
			if channel == "" {
				fmt.Println("## analysis")
				channel = "analysis"
			}
			fmt.Print(delta.ReasoningContent)
		}
		if delta.Content != "" {
			if channel == "analysis" {
				fmt.Println("\n\n## final")
				channel = "final"
			}
			fmt.Print(delta.Content)
		}
	}
}
```

So that gives us everything we need to write a simple command line chat demo:

[chat.go](https://github.com/jnb666/gpt-go/tree/main/cmd/chat/chat.go)
```go
	input := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("> ")
		question, err := input.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		req.Messages = append(req.Messages, openai.ChatCompletionMessage{Role: "user", Content: question})

		resp, err := createChatCompletionStream(context.Background(), client, req, printOutput())
		if err != nil {
			log.Fatal(err)

		}
		fmt.Println()
		req.Messages = append(req.Messages, openai.ChatCompletionMessage{Role: "assistant", Content: resp.Message.Content})
	}
```

An example run below - hit ctrlD or ctrlC to exit.

```
go run cmd/chat/chat.go
> who was the first president of the United States?
## analysis
User asks a simple factual question: "who was the first president of the United States?" The answer: George Washington. Probably want concise answer.

## final
The first president of the United States was **George Washington** (served 1789–1797).
> when was he born?
## analysis
We need to answer: George Washington born February 22, 1732. Provide concise answer.

## final
George Washington was born on **February 22, 1732**.
> and where?
## analysis
The user asks: "when was he born?" "and where?" So answer: he was born on Feb 22 1732 in Westmoreland County, Virginia (or near the town of Westmoreland). He was born on the farm known as "Mount Vernon" later? Wait, no, he was born at his father's plantation, in Westmoreland County, Virginia, near the town of "Westmoreland". He was born on a farm known as "the family plantation" in Westmoreland County, but the exact location: "the family plantation, near the town of Westmoreland, in the Colony of Virginia." Historically, his birthplace: "Mount Vernon" is his later plantation but not birthplace. The birthplace is "the family plantation in Westmoreland County." Some sources say "near the town of Westmoreland (now known as "Westmoreland, Virginia"? Actually the town is Westmoreland, but the county is Westmoreland County. The farm was called "the family plantation." Also known as "the family farm." The exact location: "the family farm in Westmoreland County, Virginia, near the town of Westmoreland." So answer: Born February 22, 1732, at his father's farm in Westmoreland County, Virginia.

## final
George Washington was born on **February 22, 1732** at his family’s farm in **Westmoreland County, Virginia** (near the town of Westmoreland, in the Colony of Virginia).
> ^C
```

In the next post we'll extend this to cover tool calling so that the model can generate function calls and process the results.

