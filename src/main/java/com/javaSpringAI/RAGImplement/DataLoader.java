package com.javaSpringAI.RAGImplement;

import jakarta.annotation.PostConstruct;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.reader.pdf.config.PdfDocumentReaderConfig;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.stereotype.Component;

@Component
public class DataLoader {
    private final VectorStore vectorStore;
    private final JdbcClient jdbcClient;

    @Value("classpath:/xyz.pdf")
    private Resource pdfResource;

    public DataLoader(VectorStore vectorStore, JdbcClient jdbcClient) {
        this.vectorStore = vectorStore;
        this.jdbcClient = jdbcClient;
    }

    @PostConstruct
    public void init(){
       Integer count = jdbcClient.sql("select COUNT(*) from vector_store").query(Integer.class).single();
        System.out.println("No of record count " + count);
        if(count == 0){
            System.out.println("Loading new record in PG database");
            PdfDocumentReaderConfig cfg = PdfDocumentReaderConfig.builder().withPagesPerDocument(1).build();
            PagePdfDocumentReader reader = new PagePdfDocumentReader(pdfResource, cfg);
            var textSplitter = new TokenTextSplitter();
            vectorStore.accept(textSplitter.apply(reader.get()));
            System.out.println("Loaded new record in PG database successfully");
        }
    }
}
