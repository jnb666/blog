+++
date = '2025-09-05T13:06:14+01:00'
title = 'Local LLM models: Part 5 - adding a web browser tool'
+++

In this final post of the series we'll add tool functions which will allow the model to retrieve new information by 
performing web searches and parsing the results. 
We'll use a similar interface to the one used by OpenAI in the training of the gpt-oss models.
See the [OpenAI github](https://github.com/openai/gpt-oss?tab=readme-ov-file#tools) for their sample implementation in
Python. 

So we need to add 3 tool functions which will implement the same `ToolFunction` interface we defined in
[part 3](/posts/local_llm_hosting-part3).

- **browser.search** - Performs a web search and returns a list of links with title and description for each.
- **browser.open** - Opens a URL and returns an extract from the page content in Markdown format.
- **browser.find** - Searches for a literal string within an opened page.

The code described below is a reimplentation in go, using the [Brave search API](https://brave.com/search/api/)
for web search and a local [Firecrawl](https://www.firecrawl.dev/) instance to scape web pages and convert the content
to Markdown.

<!--more-->

### Defining the browser tools

First we'll extend the existing [markdown](https://github.com/jnb666/gpt-go/blob/main/markdown/markdown.go) package
to add a `Document` type for the retrieved web page contents.

```go
type Document struct {
	Title      string
	URL        string
	Links      []Link
	Lines      []string
	StartLine  int
	WrapColumn int
}

type Link struct {
	Title string
	URL   string
}
```

Then create a new browser package and define the `Browser` type to hold the list of retrieved
documents and the cursor which indicates the current document. There may be multiple tool calls
for each round of the chat so we need to retain the retrieved documents for use in subsequent calls.
When the next message from the user is submitted the saved state can be reset.

[api/tools/browser/browser.go](https://github.com/jnb666/gpt-go/blob/main/api/tools/browser/browser.go).
```go
type Browser struct {
	BraveApiKey     string
	FirecrawlApiKey string
	Docs            []markdown.Document
	Cursor          int
}

func (b *Browser) Tools() []api.ToolFunction {
	return []api.ToolFunction{
		&Search{Browser: b, MaxWords: MaxWords},
		&Open{Browser: b, MaxWords: MaxWords},
		&Find{Browser: b, MaxWords: FindMaxWords},
	}
}

func (b *Browser) Reset() {
	b.Cursor = 0
	b.Docs = b.Docs[:0]
}

func (b *Browser) Description() string {
	return "## browser\n\n" +
		"// Tool for browsing.\n" +
		"// The `cursor` appears in brackets before each browsing display: `[{cursor}]`.\n" +
		"// Cite information from the tool using the following format:\n" +
		"// `【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.\n" +
		"// Do not quote more than 10 words directly from the tool output."
}

func (b *Browser) current(cursor int) (doc markdown.Document, err error) {
	if cursor >= 0 && cursor < len(b.Docs) {
		return b.Docs[cursor], nil
	}
	if b.Cursor >= 0 && b.Cursor < len(b.Docs) {
		return b.Docs[b.Cursor], nil
	}
	return doc, fmt.Errorf("Document at cursor %d not found", cursor)
}

func (b *Browser) add(doc markdown.Document) {
	b.Cursor = len(b.Docs)
	b.Docs = append(b.Docs, doc)
}
```

For each tool function we need to add the Definition method with the description and schema definition.
These are provided to the model in the developer message.

```go
type Search struct {
	*Browser
	MaxWords int
}

func (t Search) Definition() openai.Tool {
	fn := openai.FunctionDefinition{
		Name:        "browser.search",
		Description: "Searches the web for information related to `query` and displays `topn` results.",
		Parameters: json.RawMessage(`{
	"type": "object",
	"properties": {
		"query": {"type":"string"},
		"topn": {"type":"number", "description":"default: 10"}
	},
	"required": ["query"]
}`)}
	return openai.Tool{Type: openai.ToolTypeFunction, Function: &fn}
}
```

```go
type Open struct {
	*Browser
	MaxWords int
}

func (t Open) Definition() openai.Tool {
	fn := openai.FunctionDefinition{
		Name: "browser.open",
		Description: "Opens the link `id` from the page indicated by `cursor` starting at line number `loc`.\n" +
			"Valid link ids are displayed with the formatting: `【{id}†.*】`.\n" +
			"If `cursor` is not provided, the most recent page is implied.\n" +
			"If `id` is a string, it is treated as a fully qualified URL.\n" +
			"If `loc` is not provided, the viewport will be positioned at the beginning of the document.\n" +
			"Use this function without `id` to scroll to a new location of an opened page.",
		Parameters: json.RawMessage(`{
	"type": "object",
	"properties": {
		"cursor": {"type": "number", "description":"default: -1"},
		"id": {"type":["number","string"], "description":"default: -1"},
		"loc": {"type":"number", "description":"default: -1"}
	}
}`)}
	return openai.Tool{Type: openai.ToolTypeFunction, Function: &fn}
}
```

```go
type Find struct {
	*Browser
	MaxWords int
}

func (t Find) Definition() openai.Tool {
	fn := openai.FunctionDefinition{
		Name:        "browser.find",
		Description: "Finds exact matches of `pattern` in the current page, or the page given by `cursor`.",
		Parameters: json.RawMessage(`{
	"type": "object",
	"properties": {
		"pattern": {"type":"string"},
		"cursor": {"type":"number", "description":"default: -1"}
	},
	"required": ["pattern"]
}`)}
	return openai.Tool{Type: openai.ToolTypeFunction, Function: &fn}
}
```

### Generating the system prompt

If you examine the [chat template](/docs/jinja_chat_template) you can see that the definitions given above are
based on those for the builtin browser tool with a few extra arguments omitted.

So why not just set the `builtin_tools` field in `chat_template_kwargs` to use that? 
Because as well as generating the prompt we also need llama.cpp to parse the returned tool calls. This depends on
adding the functions to the `tools` array. Defining both does work but means that the info is duplicated in the prompt, 
so instead I have omitted this.

To generate the `openai.ChatCompletionRequest` which we send to the completions endpoint the code looks something like:

```go
	// initialisation
	browse := &browser.Browser{BraveApiKey: os.Getenv("BRAVE_API_KEY")}
	tools := browse.Tools()
	cfg := api.DefaultConfig(tools...)
	cfg.ToolDescription = browse.Description()

	// start chat
	conv := api.NewConversation(cfg)
	conv.Messages = append(conv.Messages, api.Message{Type: "user", Content: "<user input>"})

	req := api.NewRequest(conv, tools...)
```

I've added a new ToolDescription field to the `api.Config` struct which was defined in the part 4 of this series.
This is used by `api.NewRequest` to add the shared description to the developer message before the Tools section.


### Dependencies

#### 1. Brave web search

I've chosen to use the [Brave search API](https://brave.com/search/api/) for the web search call as they provide an easy 
to use Rest API and offer a free tier with limits of 2000 req/month and 1 req/sec. 
You'll need to register on their site to get an API key.
The [web search API docs](https://api-dashboard.search.brave.com/app/documentation/web-search/get-started)
are on their website. 

Here's the code which calls this from the browser module where 
`tools.Get` is defined in [api/tools/tools.go](https://github.com/jnb666/gpt-go/blob/main/api/tools/tools.go).
It performs the HTTP GET request and unmarshals the JSON response.

```go
var (
	Country            = "gb"
	Language           = "en"
	BraveSearchURL     = "https://api.search.brave.com/res/v1/web/search"
)

type searchResponse struct {
	Web struct {
		Results []struct {
			Title, URL, Description string
		}
	}
}

func (t Search) search(query string, topn int) (resp searchResponse, err error) {
	if t.BraveApiKey == "" {
		return resp, fmt.Errorf("BraveApiKey is required for search")
	}
	uri := fmt.Sprintf("%s?q=%s&count=%d&country=%s&search_lang=%s&text_decorations=false",
		BraveSearchURL, url.QueryEscape(query), topn, Country, Language)
	err = tools.Get(uri, &resp, tools.Header{Key: "X-Subscription-Token", Value: t.BraveApiKey})
	if err != nil {
		return resp, err
	}
	if len(resp.Web.Results) == 0 {
		return resp, fmt.Errorf("no search results returned")
	}
	return resp, nil
}
```

#### 2. Firecrawl scrape API

We also need some way to retreive the web page, extract the useful info from this and return it as a Markdown document
which can be ingested by the model. To simplify things I'm using an existing tool for this rather than building something
from scratch (which would be a whole other series of blog posts!).

There are many companies who provide hosted services for this. [Firecrawl](https://firecrawl.dev) provide a paid hosted service where the
**/scape** endpoint can be called for a single URL to get the parsed content. See [their API docs](https://docs.firecrawl.dev/features/scrape) 
for the full details.

They also provide source code at [https://github.com/firecrawl/firecrawl](https://github.com/firecrawl/firecrawl) 
so we can run our own firecrawl server locally.
This does not provide all of the features of the hosted version, but gives us everything needed for this example.
See their [docs here](https://github.com/firecrawl/firecrawl/blob/main/SELF_HOST.md) for an overview of how to setup. 
Note that if not using the docker build you may also need to compile the shared libs as per 
[Dockerfile](https://github.com/firecrawl/firecrawl/blob/main/apps/api/Dockerfile), and if running the server and
workers on the same host set the PORT env variable for each.

This is the code which generates a Firecrawl request and parses the response. In case the title is not returned in the page metadata
then the `htmlTitle` func gets it from the `<title>` HTML tag.

```go
var (
	Timeout            = 5 * time.Second
	FirecrawlURL       = "http://localhost:3002/v2/scrape"
	MaxCacheAge        = 24 * time.Hour
)

type scrapeResponse struct {
	Success bool
	Data    struct {
		Markdown string
		RawHTML  string
		Metadata struct {
			Title      string
			StatusCode int
		}
	}
}

func (t Open) scrape(url, title string) (doc markdown.Document, err error) {
	request := map[string]any{
		"url":     url,
		"formats": []string{"markdown", "rawHtml"},
		"maxAge":  MaxCacheAge.Milliseconds(),
		"timeout": Timeout.Milliseconds(),
	}
	var reply scrapeResponse
	log.Info("    ", url)
	err = tools.Post(FirecrawlURL, request, &reply, tools.Header{Key: "Authorization", Value: "Bearer " + t.FirecrawlApiKey})
	if err != nil {
		return doc, err
	}
	if reply.Data.Metadata.Title != "" {
		title = reply.Data.Metadata.Title
	} else {
		title = htmlTitle(reply.Data.RawHTML, title, url)
	}
	if !reply.Success {
		return doc, fmt.Errorf("error retrieving page: %s", title)
	}
	doc = markdown.QuoteLinks(reply.Data.Markdown, url, title, WrapColumn)
	return doc, err
}
```

The `markdown.quoteLinks` function parses the Markdown content to extract the links, adds them to the returned `Document.Links` array
and replaces them in the text with either 【{index}†{title}†{host}】for offsite links or【{index}†{title}】for links to the same site.

My first attempt at this was using regexes to transform the content, but that gets messy pretty fast. So instead
this is implemented with the [goldmark](https://github.com/yuin/goldmark) parser which we already using to render Markdown
content to HTML. We can inject a custom AST transformer to rewrite any Link nodes and remove Image nodes, then use
[goldmark-markdown](https://github.com/teekennedy/goldmark-markdown) renderer to export the document back to Markdown format.
See [markdown.go](https://github.com/jnb666/gpt-go/blob/main/markdown/markdown.go#L91) for the code.


### Tool call implementation

We now need to implement the Call method to satisfy the `api.ToolFunction` interface for each of the 3 tools:

#### browser.search

This is pretty straightforward, we just parse the arguments from JSON format, perform the search, convert the list of links 
into a markdown document, add it to the browser Docs list and return the result.

```go
func (t Search) Call(arg json.RawMessage) string {
	log.Infof("[%d] browser.search(%s)", len(t.Docs), arg)
	var args struct {
		Query string
		TopN  int
	}
	args.TopN = 10
	if err := json.Unmarshal(arg, &args); err != nil {
		return errorResponse(err)
	}
	if strings.TrimSpace(args.Query) == "" {
		return errorResponse(fmt.Errorf("query argument is required"))
	}
	resp, err := t.search(args.Query, args.TopN)
	if err != nil {
		return errorResponse(err)
	}
	doc := markdown.Document{
		Title:      fmt.Sprintf("Web search for “%s”", args.Query),
		URL:        fmt.Sprintf("https://search.brave.com/search?q=%s&source=web", url.QueryEscape(args.Query)),
		WrapColumn: WrapColumn,
	}
	doc.Write("# Search Results\n\n")
	for i, r := range resp.Web.Results {
		link := markdown.Link{URL: r.URL, Title: r.Title}
		ref := link.Format(i, "")
		log.Debugf("%s %s", ref, r.URL)
		doc.Write("  * " + ref + "\n" + r.Description + "\n")
		doc.Links = append(doc.Links, link)
	}
	t.add(doc)
	return doc.Format(t.Cursor, t.MaxWords)
}
```

There are few helpers defined in [markdown.go](https://github.com/jnb666/gpt-go/blob/main/markdown/markdown.go):
- `link.Format` formats a link as【{index}†{title}†{host}】.
- `doc.Write` appends text to the document, wrapping long lines if needed.
- `doc.Format` returns the document with Title and URL heading and content with line numbers added. 

Returned content will start at StartLine (0 in this case) and is truncated at MaxWords.

`browser_test.go` contains some simple smoke tests to check this works and dump the response and the extracted links:

```
~/gpt-go/api/tools/browser$ go test -v -run Search
=== RUN   TestSearch
INFO[0000] [0] browser.search({"query":"local LLM hosting"})
    browser_test.go:11: response:
        [0] Web search for “local LLM hosting”
        (https://search.brave.com/search?q=local+LLM+hosting&source=web)
        **viewing lines [1 - 28] of 43**

        L1: # Search Results
        L2:
        L3:   * 【0†r/LocalLLaMA on Reddit: Current best options for local LLM hosting?】
        L4: 64 votes, 39 comments. Per the title, I’m looking to host a small finetuned LLM on my local hardware. I would like to make
        L5: it accessible via API to…
        L6:
        L7:   * 【1†How to Run a Local LLM: Complete Guide to Setup & Best Models (2025) – n8n Blog】
        L8: While there might be upfront costs ... significant savings in the long run. This makes local LLMs a more cost-effective
        L9: solution, especially for high-volume usage. ... Looking for the fastest way to build your own self-hosted AI workflows?...
        L10:
        L11:   * 【2†I started self-hosting LLMs and absolutely loved it】
        L12: With how fast local models have improved, I wanted to see if running my own LLM was finally practical. There are several
        L13: reasons to try hosting LLMs yourself, so I gave self-hosting a shot, and I was actually pretty amazed.
        L14:
        L15:   * 【3†6 Self-Hosted & Local LLMs】
        L16: A self-hosted LLM is a large language model that runs on your own hardware or infrastructure. This could be a local computer,
        L17: server, or edge device, or using a containerization service like Docker or Kubernetes.
        L18:
        L19:   * 【4†Building a Low-Cost Local LLM Server to Run 70 Billion Parameter Models】
        L20: The final step is to install OLLAMA locally and test it with your configured models. ... Use Homebrew to install OLLAMA,
        L21: then download and configure your LLM model. ... brew install ollama export OLLAMA_HOST=http://localhost:3000 # This should
        L22: return the models from the localhost:3000 ollama list
        L23:
        L24:   * 【5†Run LLMs Locally: 7 Simple Methods | DataCamp】
        L25: Run LLMs locally (Windows, macOS, Linux) by leveraging these easy-to-use LLM frameworks: GPT4All, LM Studio, Jan, llama.cpp,
        L26: llamafile, Ollama, and NextChat.
        L27:
        L28:   * 【6†Self-Hosted LLM: A 5-Step Deployment Guide】
    browser_test.go:76: 0: https://www.reddit.com/r/LocalLLaMA/comments/1767pyg/current_best_options_for_local_llm_hosting/
    browser_test.go:76: 1: https://blog.n8n.io/local-llm/
    browser_test.go:76: 2: https://www.xda-developers.com/i-started-self-hosting-llms-absolutely-loved-it/
    browser_test.go:76: 3: https://budibase.com/blog/ai-agents/local-llms/
    browser_test.go:76: 4: https://www.comet.com/site/blog/build-local-llm-server/
    browser_test.go:76: 5: https://www.datacamp.com/tutorial/run-llms-locally-tutorial
    browser_test.go:76: 6: https://www.plural.sh/blog/self-hosting-large-language-models/
    browser_test.go:76: 7: https://www.docker.com/blog/llm-docker-for-local-and-hugging-face-hosting/
    browser_test.go:76: 8: https://www.linkedin.com/pulse/ultimate-guide-hosting-your-own-local-large-language-model-ingram-l5jof
    browser_test.go:76: 9: https://medium.com/thedeephub/50-open-source-options-for-running-llms-locally-db1ec6f5a54f
--- PASS: TestSearch (1.09s)
PASS
ok  	github.com/jnb666/gpt-go/api/tools/browser	1.100s
```

#### browser.open

When we parse the arguments there are three possible cases for the returned document:

1. `id` is a string - return a new document from scraping this URL.
2. `id` is a number - return a new document from scraping link number `id` in the document given by `cursor`.
3. `id` not set - return document given by `cursor`.

If `cursor` is omitted the most current page is used. If `loc` is set then the start line number is set. 
We always add a new document to the Browser Docs list. This might be either the new scraped page or a 
copy of a previous one scrolled to a new start line.

```go
func (t Open) Call(arg json.RawMessage) string {
	log.Infof("[%d] browser.open(%s)", len(t.Docs), arg)
	var args struct {
		Cursor int
		ID     any
		Loc    int
	}
	args.Cursor = -1
	args.Loc = -1
	if err := json.Unmarshal(arg, &args); err != nil {
		return errorResponse(err)
	}
	var current, doc markdown.Document
	var err error
	switch url := args.ID.(type) {
	case string:
		doc, err = t.scrape(url, "")
	case float64:
		id := int(url)
		if current, err = t.current(args.Cursor); err == nil {
			if id >= 0 && id < len(current.Links) {
				doc, err = t.scrape(current.Links[id].URL, current.Links[id].Title)
			} else {
				doc = current
			}
		}
	default:
		doc, err = t.current(args.Cursor)
	}
	if err != nil {
		return errorResponse(err)
	}
	if args.Loc >= 0 {
		doc.StartLine = args.Loc
	}
	t.add(doc)
	return doc.Format(t.Cursor, t.MaxWords)
}
```

To check that is working:

```
~/gpt-go/api/tools/browser$ go test -v -run OpenID
=== RUN   TestOpenID
INFO[0000] [0] browser.search({"query":"local LLM hosting"})
INFO[0001] [1] browser.open({"id":3})
INFO[0001]     https://budibase.com/blog/ai-agents/local-llms/
    browser_test.go:30: response:
        [1] 6 Self-Hosted & Local LLMs
        (https://budibase.com/blog/ai-agents/local-llms/)
        **viewing lines [1 - 42] of 277**

        L1: 【0†<- All posts】
        L2:
        L3: With fast-advancing technology, running AI models locally is no longer the preserve of massive enterprises or researchers.
        L4: Today, smaller businesses and even hobbyists are also leveraging self-hosted LLMs within development projects.
        L5:
        L6: Thanks to advances in model quantization, local runtimes and runners, and smaller yet highly capable models, it’s increasingly
        L7: viable to run LLMs in cloud containers or even consumer hardware.
        L8:
        L9: But, of course, this introduces a range of challenges. One of the biggest is choosing the right model for our needs.
        L10:
        L11: That is, we not only need a model that’s suitable for our use case, but also one that is self-hostable with the compute
        L12: resources we have available.
        L13:
        L14: Today, we’re diving deep into exactly how to make this decision.
        L15:
        L16: Specifically, we’ll be covering:
        L17:
        L18: * 【1†What is a self-hosted LLM?】
        L19:
        L20: * 【2†How can you run a model locally?】
        L21:
        L22: * 【3†Why would we want to host a model locally?】
        L23:
        L24: * 【4†What to look for in a self-hosted LLM】
        L25:
        L26: * 【5†6 self-hosted LLMs for 2025】
        L27:
        L28: Let’s start with the basics.
        L29:
        L30: ## What is a self-hosted LLM?
        L31:
        L32: A self-hosted LLM is a large language model that runs on your own hardware or infrastructure. This could be a local computer,
        L33: server, or edge device, or using a containerization service like Docker or Kubernetes.
        L34:
        L35: We can contrast this with commercially available models that are offered as services, where we’re reliant on the vendor’s
        L36: API.
        L37:
        L38: As such, self-hosting gives us full control over the model, environment, and data, with all inference happening locally.
        L39: We also usually won’t have to pay a usage fee for making API requests.
        L40:
        L41: Models that are suitable for self-hosting have a few key characteristics in common.
        L42:
...
    browser_test.go:76: 0: https://budibase.com/blog/
    browser_test.go:76: 1: https://budibase.com/blog/ai-agents/local-llms/#what-is-a-self-hosted-llm
    browser_test.go:76: 2: https://budibase.com/blog/ai-agents/local-llms/#how-can-you-run-a-model-locally
    browser_test.go:76: 3: https://budibase.com/blog/ai-agents/local-llms/#why-would-we-want-to-host-a-model-locally
    browser_test.go:76: 4: https://budibase.com/blog/ai-agents/local-llms/#what-to-look-for-in-a-self-hosted-llm
    browser_test.go:76: 5: https://budibase.com/blog/ai-agents/local-llms/#6-self-hosted-llms-for-2025
    browser_test.go:76: 6: https://budibase.com/blog/ai-agents/enterprise-chatbots/
    browser_test.go:76: 7: https://budibase.com/blog/ai-agents/open-source-llms/
    browser_test.go:76: 8: https://budibase.com/blog/ai-agents/local-llms/#1-mistral-7b
    browser_test.go:76: 9: https://budibase.com/blog/ai-agents/local-llms/#2-phi-3-mini
    browser_test.go:73: ...
--- PASS: TestOpenID (0.94s)
PASS
ok  	github.com/jnb666/gpt-go/api/tools/browser	0.947s
```

#### browser.find

The find tool is similar to calling open with a null id. We get the document given by the `cursor`
parameter, or the current document if not set. Then search for `pattern` in the text and
update the `StartLine` field and the page title.
If called more than once it will search for the next occurance of the search string starting at the line
following the previous match.

```go
var reFind = regexp.MustCompile(`^Find results for “(.+?)” in “(.+?)”`)

func (t Find) Call(arg json.RawMessage) string {
	log.Infof("[%d] browser.find(%s)", len(t.Docs), arg)
	var args struct {
		Pattern string
		Cursor  int
	}
	args.Cursor = -1
	if err := json.Unmarshal(arg, &args); err != nil {
		return errorResponse(err)
	}
	if strings.TrimSpace(args.Pattern) == "" {
		return errorResponse(fmt.Errorf("pattern argument is required"))
	}
	current, err := t.current(args.Cursor)
	if err != nil {
		return errorResponse(err)
	}
	if m := reFind.FindStringSubmatch(current.Title); len(m) > 0 {
		current.Title = m[2]
		if m[1] == args.Pattern {
			current.StartLine++
		}
	}
	line := current.Find(args.Pattern)
	doc := current
	if line >= 0 {
		doc.Title = fmt.Sprintf("Find results for “%s” in “%s”", args.Pattern, current.Title)
		doc.StartLine = line
	} else {
		doc.Title = fmt.Sprintf("“%s” not found in page “%s”", args.Pattern, current.Title)
		doc.StartLine = len(doc.Lines)
	}
	t.add(doc)
	return doc.Format(t.Cursor, t.MaxWords)
}
```

### Converting references back to links

In the final response from the model any links referenced will be in 【{cursor}†L{line_start}(-L{line_end})?】format
as per the instructions we provided in the prompt. To make these more useful I've added a further post-processing step
to convert them back to Markdown links with the URL. This can then be called after the final response is returned.

```go
var citationRegexp = regexp.MustCompile(`【(\d+)†(L.+?)】`)

func (s *Browser) Postprocess(content string) string {
	return citationRegexp.ReplaceAllStringFunc(content, func(ref string) string {
		m := citationRegexp.FindStringSubmatch(ref)
		if len(m) != 3 {
			log.Warnf("postprocess: citation regex match failed for %q", ref)
			return ref
		}
		cursor, err := strconv.Atoi(m[1])
		if err != nil || cursor < 0 || cursor >= len(s.Docs) {
			log.Warnf("postprocess: error parsing citation %q - invalid cursor", ref)
			return ref
		}
		url := s.Docs[cursor].URL
		title := linkTitle(s.Docs[cursor].Title)
		log.Debugf("parse citation %q cursor=%d lines=%s url=%s", ref, cursor, m[2], url)
		return fmt.Sprintf(" [%s†%s](%s %q) ", markdown.URLHost(url), m[2], url, title)
	})
}
```

I've also added a small tweak to the `markdown.Render` function to add `target="_blank"` attribute so that links open 
in a new browser tab.


### Testing it all works

To check this with a simple query which requires current knowledge you can run the
[browser_tool.go](https://github.com/jnb666/gpt-go/blob/main/cmd/browser_tool/browser_tool.go) command line test.

Example output:

```
~/gpt-go$ go run cmd/browser_tool/browser_tool.go
## analysis
The user asks: "who is Prime Minister of the UK?" We need to provide current PM. As of 2025, UK PM is Rishi Sunak? Wait: Rishi Sunak became PM in October 2022, but then in 2024? Actually Rishi Sunak is still PM as of 2025. So answer: Rishi Sunak. But we should browse for confirmation.

## tool call
INFO[0000] [0] browser.search({"query":"current prime minister of the United Kingdom 2025","topn":5})

## tool response
[0] Web search for “current prime minister of the United Kingdom 2025”
(https://search.brave.com/search?q=current+prime+minister+of+the+United+Kingdom+2025&source=web)
**viewing lines [1 - 25] of 25**

L1: # Search Results
L2:
L3:   * 【0†Prime Minister of the United Kingdom - Wikipedia】
L4: As the leader of the world's sixth largest economy, the prime minister holds significant domestic and international leadership,
L5: being the leader of a prominent member state of NATO, the G7 and G20. As of 2025 58 people (55 men and 3 women) have served
L6: as prime minister, the first of whom was ...
L7:
L8:   * 【1†Keir Starmer - Wikipedia】
L9: Sir Keir Rodney Starmer (born 2 September 1962) is a British politician and lawyer who has served as Prime Minister of the
L10: United Kingdom since 2024 and as Leader of the Labour Party since 2020. He previously served as Leader of the Opposition
L11: from 2020 to 2024.
L12:
L13:   * 【2†Prime Minister - GOV.UK】
L14: Sir Keir Starmer became Prime Minister on 5 July 2024.
L15:
L16:   * 【3†Who would be the best prime minister? May 2025 | YouGov】
L17: However, the extent to which Nigel Farage is able to coax Tories over to Reform at the next election will be key to his
L18: party’s prospects, and the results show that almost as many of the Conservatives’ 2024 electorate would prefer Farage to
L19: be prime minister (33%) as Badenoch (38%). The ...
L20:
L21:   * 【4†UK best prime minister poll 2025| Statista】
L22: As of August 2025, 30 percent of people in Great Britain thought that Keir Starmer was better for the job of Prime Minister
L23: than the Leader of the Opposition, Kemi Badenoch, with 20 percent believing she would be better for the job. Although Starmer
L24: is currently seen as better suited for the ...
L25:

Result shows Keir Starmer is PM since July 2024. That means my earlier assumption is wrong. So need to provide correct answer. Let's open result 2.

## tool call
INFO[0002] [1] browser.open({"cursor":0,"id":2})
INFO[0002]     https://www.gov.uk/government/ministers/prime-minister

## tool response
[1] Prime Minister - GOV.UK
(https://www.gov.uk/government/ministers/prime-minister)
**viewing lines [1 - 56] of 150**

L1: ## Cookies on GOV.UK
L2:
L3: We use some essential cookies to make this website work.
L4:
L5: We’d like to set additional cookies to understand how you use GOV.UK, remember your settings and improve government services.
L6:
L7: We also use cookies set by other sites to help us deliver content from their services.
L8:
L9: You have accepted additional cookies. You can 【0†change your cookie settings】at any time.
L10:
L11: You have rejected additional cookies. You can 【1†change your cookie settings】at any time.
L12:
L13: Accept additional cookies Reject additional cookies 【2†View cookies】
L14:
L15: Hide this message
L16:
L17: 【3†Skip to main content】
L18:
L19: 【4†Emergency Alerts】Test on Sunday 7 September, 3pm
L20:
L21: Organisations: 【5†Cabinet Office】and 【6†Prime Minister's Office, 10 Downing Street】
L22:
L23: Current role holder: 【7†The Rt Hon Sir Keir Starmer KCB KC MP】
L24:
L25: ## Responsibilities
L26:
L27: The Prime Minister is the leader of His Majesty’s Government and is ultimately responsible for the policy and decisions
L28: of the government.
L29:
L30: As leader of the UK government the Prime Minister also:
L31:
L32: * oversees the 【8†operation of the Civil Service】 and government agencies
L33: * chooses members of the government
L34: * is the principal government figure in the House of Commons
L35:
L36: ## Current role holder
L37:
L38: ### The Rt Hon Sir Keir Starmer KCB KC MP
L39:
L40: Sir Keir Starmer became Prime Minister on 5 July 2024.
L41:
L42: ## Education
L43:
L44: Keir attended Reigate Grammar School, before studying Law at the University of Leeds. He went on to do postgraduate studies
L45: at the University of Oxford, receiving a Bachelor of Civil Law (BCL) degree.
L46:
L47: ## Political Career
L48:
L49: Keir was elected a Member of Parliament for Holborn and St Pancras in May 2015. He was elected leader of the Labour Party
L50: in April 2020.
L51:
L52: ## Career before politics
L53:
L54: Keir qualified as a barrister in 1987 working in private practice before working as a legal advisor to the Northern Ireland
L55: Policing Board.
L56:

Lines 23-41 show current role holder is Sir Keir Starmer, became PM on 5 July 2024. Good.

Thus answer: Sir Keir Starmer.

Provide citation.

## final
The current Prime Minister of the United Kingdom is **Sir Keir Starmer**【1†L23-L41】. He assumed office on 5 July 2024.

## postprocesssed
The current Prime Minister of the United Kingdom is **Sir Keir Starmer** [www.gov.uk†L23-L41](https://www.gov.uk/government/ministers/prime-minister "Prime Minister   GOV UK") . He assumed office on 5 July 2024.
```


### Adding the browser tool to the web app

To add the browser tool to the [webchat.go](https://github.com/jnb666/gpt-go/blob/main/cmd/webchat/webchat.go) app
we developed in [part 4](/posts/local_llm_hosting-part4) there are a few small changes.

1. Add the Browser instance to the connection struct and initialise it in `newConnection`.

```go
	if apiKey := os.Getenv("BRAVE_API_KEY"); apiKey != "" {
		c.browser = &browser.Browser{BraveApiKey: apiKey}
		c.tools = append(c.tools, c.browser.Tools()...)
	} else {
		log.Warn("skipping browser tools support - BRAVE_API_KEY env variable is not defined")
	}
```

2. Add the tool description to the default config in `websocketHandler`.

```go
	cfg := api.DefaultConfig(c.tools...)
	if c.browser != nil {
		cfg.ToolDescription = c.browser.Description()
	}
```

3. Update `addMessage` to reset the browser state before each chat round and apply the `browser.Postprocess` method to
the final chat.

```go
	if c.browser != nil {
		c.browser.Reset()
	}
	_, err := api.CreateChatCompletionStream(context.Background(), c.client, req, c.streamMessage, c.tools...)
	if err != nil {
		return conv, err
	}
	if c.browser != nil && len(c.browser.Docs) > 0 {
		c.content = c.browser.Postprocess(c.content)
	}
	err = c.sendUpdate("final", "\n")
	if err != nil {
		return conv, err
	}
	if c.analysis != "" {
		conv.Messages = append(conv.Messages, api.Message{Type: "analysis", Content: c.analysis})
	}
	conv.Messages = append(conv.Messages, api.Message{Type: "final", Content: c.content})
```

Here's the screen recording for a sample run.

{{< rawhtml >}} 
<video width="100%" controls autoplay>
    <source src="/video/webchat.m4v" type="video/mp4">
    Your browser does not support the video tag.  
</video>
{{< /rawhtml >}}

And the corresponding logs to the console.

```
~/gpt-go$ go run cmd/webchat/webchat.go
INFO[1231] add message: "why was Liz Truss compared to a lettuce?"
INFO[1231] [0] browser.search({"query":"Liz Truss compared to a lettuce","topn":10})
INFO[1233] [1] browser.open({"cursor":0,"id":1})
INFO[1233]     https://en.wikipedia.org/wiki/Liz_Truss_lettuce
INFO[1233] [2] browser.find({"pattern":"lettuce","cursor":1})
INFO[1235] [3] browser.open({"cursor":0,"id":4})
INFO[1235]     https://www.politico.eu/article/liz-truss-says-being-compared-to-a-lettuce-was-not-funny/
INFO[1237] list saved chats: current=019919b9-75d8-7367-a678-fa2594e4b4cd
INFO[1249] add message: "what is she doing now?"
INFO[1249] [0] browser.search({"query":"Liz Truss current activities 2025","topn":10})
INFO[1250] [1] browser.open({"cursor":0,"id":0})
INFO[1250]     https://en.wikipedia.org/wiki/Liz_Truss
INFO[1251] [2] browser.find({"pattern":"MP","cursor":1})
INFO[1252] [3] browser.find({"pattern":"House of Commons","cursor":1})
INFO[1252] [4] browser.find({"pattern":"2025","cursor":1})
INFO[1253] [5] browser.find({"pattern":"currently","cursor":1})
INFO[1253] [6] browser.search({"query":"Liz Truss current activities 2024 2025 interview","topn":10})
INFO[1254] [7] browser.open({"cursor":6,"id":1})
INFO[1254]     https://en.wikipedia.org/wiki/Liz_Truss
INFO[1256] [8] browser.find({"pattern":"2024","cursor":7})
INFO[1256] [9] browser.open({"cursor":7,"loc":850})
INFO[1257] [10] browser.find({"pattern":"2024-","cursor":7})
INFO[1257] [11] browser.search({"query":"Liz Truss lost seat 2024 general election","topn":10})
INFO[1259] [12] browser.open({"cursor":11,"id":1})
INFO[1259]     https://members.parliament.uk/member/4097/electionresult
ERRO[1264] HTTP error: 408 Request Timeout
INFO[1264] [12] browser.open({"cursor":11,"id":0})
INFO[1264]     https://www.theguardian.com/politics/article/2024/jul/05/former-tory-prime-minister-liz-truss-loses-her-seat-to-labour
INFO[1265] [13] browser.search({"query":"Liz Truss 2025 activities","topn":10})
INFO[1267] [14] browser.open({"cursor":13,"id":0})
INFO[1267]     https://www.theguardian.com/politics/2025/apr/15/liz-truss-to-launch-uncensored-social-network-to-counter-mainstream-media
```

All of the source code for this example is at https://github.com/jnb666/gpt-go/tree/main.
There are lots of extra features we could add but hopefully this demonstrates how simple it is to build something useful 
in under 2000 lines of code.



