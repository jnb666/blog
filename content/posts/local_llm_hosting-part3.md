+++
date = '2025-08-29T17:08:49+01:00'
title = 'Local LLM models: Part 3 - calling tool functions'
+++
For the previous post in this series which is an intro to the completions API in go see 
[part 2](/posts/local_llm_hosting-part2).
In this post we will extend the command line chat app to add simple tool calling using the 
[Open Weather API](https://openweathermap.org/api). 

We will need to add the function schema definitions to the request that we send to the model so that it knows
what functions it can call. Then if the returned response ends with a tool call instead of a final response
we extract the parameters, call the function and add the call and the response to the message list before 
resending the request.

<!--more-->

### Harmony response format

The gpt-oss models use the [Harmony response format](https://cookbook.openai.com/articles/openai-harmony). Support for this is built 
into the chat template embedded in the model file and the llama.cpp server so we don't need to worry too much about this.
However when debugging it is useful to understand the raw data which is being sent to and from the model once we start adding our
own custom functions. 

Here is the [template for unsloth/gpt-oss-20b](/docs/jinja_chat_template) as extracted using this
[chat template editor](https://huggingface.co/spaces/CISCai/chat-template-editor). If you dig though this and compare to the OpenAI
docs then you can see how the it's applying the conversion.

### Tool function interface

So that we can reuse the same code for multiple tools we'll define an interface:
```go
type ToolFunction interface {
	Definition() openai.Tool
	Call(args json.RawMessage) string
}

```
Definition returns the function name and JSON schema for the arguments, Call parses the JSON encoded argument object
in args and returns the content text which is sent back to the model.

In the `createChatCompletionStream` function we need to add some logic to parse the tool calls which are returned
on the analysis channel:
```go
	if len(delta.ToolCalls) > 0 {
		if len(choice.Message.ToolCalls) == 0 {
			choice.Message.ToolCalls = delta.ToolCalls
		} else {
			choice.Message.ToolCalls[0].Function.Arguments += delta.ToolCalls[0].Function.Arguments
		}
	}
```

Then we can wrap this in a new function which checks if the model requested a tool call.
If so it calls the tool then adds the request and response to the context before resending.
This is in a loop as there may be several tool calls generated as part of the analysis phase before
the final response is produced.

This is all in a new package as [api/api.go](https://github.com/jnb666/gpt-go/tree/main/api/api.go).

```go
func CreateChatCompletionStream(ctx context.Context, client *openai.Client, request openai.ChatCompletionRequest,
	callback func(openai.ChatCompletionStreamChoiceDelta), tools ...ToolFunction) (choice openai.ChatCompletionChoice, err error) {

	for {
		choice, err := createChatCompletionStream(ctx, client, request, callback)
		if err != nil || len(choice.Message.ToolCalls) == 0 {
			return choice, err
		}
		resp, err := callTool(choice.Message.ToolCalls[0].Function, tools)
		if err != nil {
			return choice, err
		}
		callback(openai.ChatCompletionStreamChoiceDelta{Role: openai.ChatMessageRoleTool, Content: resp})

		request.Messages = append(request.Messages,
			openai.ChatCompletionMessage{
				Role:      openai.ChatMessageRoleAssistant,
				Content:   "<|channel|>analysis<|message|>" + choice.Message.ReasoningContent,
				ToolCalls: choice.Message.ToolCalls,
			},
			openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleTool,
				Content: resp,
			},
		)
	}
}

func callTool(fn openai.FunctionCall, tools []ToolFunction) (string, error) {
	for _, tool := range tools {
		if tool.Definition().Function.Name == fn.Name {
			return tool.Call(json.RawMessage(fn.Arguments)), nil
		}
	}
	return "", fmt.Errorf("Error calling %q - tool function not defined", fn.Name)
}
```

### Using the Open Weather API

To use the API you'll need to get an API key by [registering here](https://home.openweathermap.org/users/sign_up). 
The functions we are using are available free, they don't require any subscription. 

There are two `ToolFunction` types I've implemented using this.

`Current` provides the `get_current_weather` function which calls the [current weather data API](https://openweathermap.org/current):

```go
type Current struct {
	ApiKey string
}

func (t Current) Definition() openai.Tool {
	fn := openai.FunctionDefinition{
		Name: "get_current_weather",
		Description: `Get the current weather in a given location.
Returns conditions with temperatures in Celsius and wind speed in meters/second.`,
		Parameters: json.RawMessage(`{
	"type": "object",
	"properties": {
		"location": {
			"type": "string",
			"description": "The city name and ISO 3166 country code - e.g. \"London,GB\" or \"New York,US\""
		}
	},
	"required": ["location"]
}`)}
	return openai.Tool{Type: openai.ToolTypeFunction, Function: &fn}
}

func (t Current) Call(arg json.RawMessage) string {
	log.Printf("call get_current_weather(%s)", arg)
	var args struct {
		Location string
	}
	if err := json.Unmarshal(arg, &args); err != nil {
		return errorResponse(err)
	}
	w, err := currentWeather(args.Location, t.ApiKey)
	if err != nil {
		return errorResponse(err)
	}
	return w.String()
}
```

`Forecast` is similar except Name is `get_weather_forecast` and there is an optional `periods` parameter to
set the number of 3 hour forecast periods. See the [5 day weather forecast API](https://openweathermap.org/forecast5)
docs for details.

The rest of the code is pretty straightforward - it's split out into a seperate package. See 
[api/tools/weather/weather.go](https://github.com/jnb666/gpt-go/tree/main/api/tools/weather/weather.go) for the source.


### Adding the tools to our chat app

We've now got everything we need to extend the [chat.go](https://github.com/jnb666/gpt-go/tree/main/cmd/chat/chat.go) progran
we wrote in the last post to add tool calls.
Here's the complete code which is also at [cmd/tools/tools.go](https://github.com/jnb666/gpt-go/tree/main/cmd/tools/tools.go)

You'll need to set the `OWM_API_KEY` env variable with your API key from https://home.openweathermap.org/api_keys


```go
// Command line chat example with tool calling
package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"

	"github.com/jnb666/gpt-go/api"
	"github.com/jnb666/gpt-go/api/tools/weather"
	"github.com/sashabaranov/go-openai"
)

var apiKey = os.Getenv("OWM_API_KEY")

func main() {
	log.SetFlags(0)

	config := openai.DefaultConfig("")
	config.BaseURL = "http://localhost:8080/v1"
	client := openai.NewClientWithConfig(config)

	currentWeather := weather.Current{ApiKey: apiKey}
	weatherForecast := weather.Forecast{ApiKey: apiKey}

	req := openai.ChatCompletionRequest{
		Tools: []openai.Tool{currentWeather.Definition(), weatherForecast.Definition()},
	}
	input := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("> ")
		question, err := input.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		req.Messages = append(req.Messages, openai.ChatCompletionMessage{Role: "user", Content: question})

		resp, err := api.CreateChatCompletionStream(context.Background(), client, req,
			printOutput(), currentWeather, weatherForecast)
		if err != nil {
			log.Fatal(err)

		}
		fmt.Println()
		req.Messages = append(req.Messages, openai.ChatCompletionMessage{Role: "assistant", Content: resp.Message.Content})
	}
}

// Print output from chat completion stream to stdout
func printOutput() func(openai.ChatCompletionStreamChoiceDelta) {
	channel := ""
	return func(delta openai.ChatCompletionStreamChoiceDelta) {
		if delta.Role == "tool" {
			fmt.Printf("\n## tool response\n%s\n", delta.Content)
			return
		}
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
		if len(delta.ToolCalls) > 0 && delta.ToolCalls[0].Function.Name != "" {
			fmt.Println("\n\n## tool call")
			return
		}
	}
}

```
That seems to work :-)

```
$ go run cmd/tools/tools.go
> hi
## analysis
User says "hi". We should respond politely. No function calls needed.

## final
Hello! How can I help you today?
> what's the weather like in London?
## analysis
User wants current weather in London. Use function get_current_weather.

## tool call
call get_current_weather({"location":"London,GB"})

## tool response
Current weather for London,GB: 21°C - overcast clouds, wind 4.6m/s


## final
It’s currently 21 °C in London with overcast clouds and a gentle wind of about 4.6 m/s. Feel free to let me know if you’d like a forecast or anything else!
> is it going to rain this weekend?
## analysis
The user asks: "is it going to rain this weekend?" They want weather forecast. We should call get_weather_forecast for London, maybe for next 48 hours (weekend). Use the function.

## tool call
call get_weather_forecast({"location":"London,GB","periods":48})

## tool response
Weather forecast for London,GB:
- Fri, 29 Aug 2025 19:00:00: 20°C - light rain, wind 3.9m/s
- Fri, 29 Aug 2025 22:00:00: 18°C - light rain, wind 4.6m/s
- Sat, 30 Aug 2025 01:00:00: 15°C - scattered clouds, wind 3.1m/s
- Sat, 30 Aug 2025 04:00:00: 14°C - clear sky, wind 2.7m/s
- Sat, 30 Aug 2025 07:00:00: 13°C - few clouds, wind 2.8m/s
- Sat, 30 Aug 2025 10:00:00: 18°C - scattered clouds, wind 4.2m/s
- Sat, 30 Aug 2025 13:00:00: 21°C - scattered clouds, wind 6.2m/s
- Sat, 30 Aug 2025 16:00:00: 21°C - overcast clouds, wind 6.4m/s
- Sat, 30 Aug 2025 19:00:00: 18°C - light rain, wind 6.1m/s
- Sat, 30 Aug 2025 22:00:00: 18°C - light rain, wind 6.4m/s
- Sun, 31 Aug 2025 01:00:00: 18°C - light rain, wind 5.2m/s
- Sun, 31 Aug 2025 04:00:00: 15°C - light rain, wind 3.9m/s
- Sun, 31 Aug 2025 07:00:00: 14°C - broken clouds, wind 2.8m/s
- Sun, 31 Aug 2025 10:00:00: 18°C - broken clouds, wind 5.2m/s
- Sun, 31 Aug 2025 13:00:00: 21°C - light rain, wind 6.1m/s
- Sun, 31 Aug 2025 16:00:00: 21°C - overcast clouds, wind 7.0m/s
- Sun, 31 Aug 2025 19:00:00: 19°C - broken clouds, wind 4.5m/s
- Sun, 31 Aug 2025 22:00:00: 17°C - few clouds, wind 3.3m/s
- Mon, 01 Sep 2025 01:00:00: 16°C - few clouds, wind 4.0m/s
- Mon, 01 Sep 2025 04:00:00: 15°C - scattered clouds, wind 4.4m/s
- Mon, 01 Sep 2025 07:00:00: 15°C - scattered clouds, wind 5.5m/s
- Mon, 01 Sep 2025 10:00:00: 18°C - light rain, wind 6.7m/s
- Mon, 01 Sep 2025 13:00:00: 20°C - light rain, wind 7.7m/s
- Mon, 01 Sep 2025 16:00:00: 19°C - light rain, wind 6.7m/s
- Mon, 01 Sep 2025 19:00:00: 17°C - light rain, wind 6.2m/s
- Mon, 01 Sep 2025 22:00:00: 15°C - light rain, wind 5.5m/s
- Tue, 02 Sep 2025 01:00:00: 15°C - overcast clouds, wind 5.4m/s
- Tue, 02 Sep 2025 04:00:00: 14°C - scattered clouds, wind 4.1m/s
- Tue, 02 Sep 2025 07:00:00: 14°C - scattered clouds, wind 3.0m/s
- Tue, 02 Sep 2025 10:00:00: 18°C - clear sky, wind 4.2m/s
- Tue, 02 Sep 2025 13:00:00: 20°C - light rain, wind 5.5m/s
- Tue, 02 Sep 2025 16:00:00: 16°C - light rain, wind 3.1m/s
- Tue, 02 Sep 2025 19:00:00: 16°C - light rain, wind 1.3m/s
- Tue, 02 Sep 2025 22:00:00: 14°C - broken clouds, wind 1.6m/s
- Wed, 03 Sep 2025 01:00:00: 14°C - broken clouds, wind 2.2m/s
- Wed, 03 Sep 2025 04:00:00: 13°C - overcast clouds, wind 2.3m/s
- Wed, 03 Sep 2025 07:00:00: 14°C - overcast clouds, wind 2.5m/s
- Wed, 03 Sep 2025 10:00:00: 15°C - light rain, wind 3.9m/s
- Wed, 03 Sep 2025 13:00:00: 16°C - light rain, wind 6.0m/s
- Wed, 03 Sep 2025 16:00:00: 18°C - light rain, wind 5.7m/s



## final
Here’s what the forecast says for the weekend in London (Saturday – Sunday):

| Date & Time (local) | Temp | Condition | Wind |
|---------------------|------|-----------|------|
| Sat 30 Aug 19:00   | 18 °C | Light rain | 6.1 m/s |
| Sat 30 Aug 22:00   | 18 °C | Light rain | 6.4 m/s |
| Sun 31 Aug 01:00   | 18 °C | Light rain | 5.2 m/s |
| Sun 31 Aug 04:00   | 15 °C | Light rain | 3.9 m/s |
| Sun 31 Aug 07:00   | 14 °C | Broken clouds | 2.8 m/s |
| Sun 31 Aug 10:00   | 18 °C | Broken clouds | 5.2 m/s |
| Sun 31 Aug 13:00   | 21 °C | Light rain | 6.1 m/s |
| Sun 31 Aug 16:00   | 21 °C | Overcast clouds | 7.0 m/s |
| Sun 31 Aug 19:00   | 19 °C | Broken clouds | 4.5 m/s |
| Sun 31 Aug 22:00   | 17 °C | Few clouds | 3.3 m/s |

**Bottom line:**
Yes, it looks like London will experience **light rain** throughout most of Saturday and Sunday, especially from the late afternoon onward. The mornings are a mix of cloudy and scattered clouds with a few lighter showers.

If you need more details or want to adjust the forecast window, just let me know!

```

In [part 4](/posts/local_llm_hosting-part4) we take this app and replace the command line interface with a browser based front end.
