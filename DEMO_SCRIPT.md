This project uses the CLINC150 dataset to classify English user messages by intent. It compares four models, displays each prediction and confidence score, and routes uncertain or out-of-scope messages to a human agent.

I’ll start with an in-scope message: “What is the weather in Seattle?” The table shows the intent and confidence predicted by each model, along with the team that would receive the request.

Now I’ll try an out-of-scope message: “How much has the Dow changed today?” In our measured run, the INT8 model assigned it to spending_history with 24.12% confidence. Since this is below the 45% routing threshold, the app sends it to a human agent.

On the Apple M2 CPU, the INT8 model reached 84.85% accuracy and 42.00% out-of-scope recall. Its measured size was 133.31 megabytes, with a mean latency of 6.63 milliseconds per message. Out-of-scope detection remains a limitation, so the next step is to tune the routing threshold using validation data.
