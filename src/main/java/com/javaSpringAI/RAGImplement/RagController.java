package com.javaSpringAI.RAGImplement;

import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.chat.prompt.SystemPromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

@RestController
public class RagController {
    private final ChatModel chatModel;
    private final VectorStore vectorStore;

    private String sysPrompt =
                """
                    You are a helpful assistant. 
                    Use only the following information to answer the question.
                    Do not use any other information. 
                    If you do not know simply answer : I dont have much information about it. Please google it out.
                    
                    QUESTION:
                    {input}
                    
                    DOCUMENTS:
                    {documents}   
                    """ ;

    public RagController(ChatModel chatModel, VectorStore vectorStore) {
        this.chatModel = chatModel;
        this.vectorStore = vectorStore;
    }
    @GetMapping("/")
    public String simplyfy(@RequestParam(value = "question", defaultValue = "List all the Contents of the Document")
                           String question){
        PromptTemplate template = new PromptTemplate(sysPrompt);
        Map<String, Object> promptParam = new HashMap<>();
        promptParam.put("input", question);
        promptParam.put("documents", findSimilarResult(question));

        return chatModel.call(template.create(promptParam)).getResult().getOutput().getContent();
    }

    private String findSimilarResult(String question) {
        List<Document> docs = vectorStore.similaritySearch(SearchRequest.query(question).withTopK(7));
        return docs.stream().map(doc -> doc.getContent().toString()).collect(Collectors.joining());
    }
}
