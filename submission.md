1.-My Understanding Of The CodeBase:
-
The main purpose is
Automatically generating documentation given a code base zip file as an input.

It integrates many features in the documentation such as import statements,API specifications,dependencies and any schema tables used.

This results in documentation that can be read and understood by anyone concerned much easier and fills the gap left by manual documentation and any inconsistencies.

My understanding of the repository is that it is split into client and server side codes which work together to seamlessly generate documentation that is user-friendly.

The front end uses javascript framework Next.js and backend uses FastAPI.

Many machine learning models like BERT, Agglomerative Clustering and GPT-3.5 are used for automatically reading the code base.Utilizing NLP and Its models to reduce manual labour is the reason for this project. 

Each one playes its own role as mentioned below:

-The BERT model for Tokenization and Code Embedding which helps us read the given codebase effectively. 

-Clustering and GPT for maintaining context and finding similarities for generation of accurate and human readable documentation.

-GPT further gives the ability to add additional features like code refactoring and test generation which makes the documentation more efficient.


2.-The Different Machine Learning Process Used In Each Step Is:
-


2.1 Code Traversal:
-
This process involves traversing the entire code and processing using simple algorithm like breadth first search(graph and nodes).Specifically, it reads Python files, processes them to extract embeddings and generate documentation using AI.

-This is further analysed using NLP models like BERT to read and convert into tokens which are embedded to convert the contextual data to numerical data which can be understood by the ML models.

-BERT uses transformer architecture which is a type of neural network used by NLP.

-It involves attention mechanism which weighs the importance between words as well as figure out relationships between them.

-After attention mechanisms, the outputs pass through neural networks for additional processing.

This ensures all words are traversed and processed and embedded properly to proceed to the next step.

These embeddings are converted to reshape all of the embeddings into a uniform format which are compatible with our clustering algorithm used later on.

These clusters use GPT-3.5 to automatically generate documentation.  


    
2.2 Code Embedding:
-

Word embedding is part of an unsupervised NLP method.It is important to capture the semantic meaning of words and undertand context.
     
The process used is BERT(Bidirectional Encoder Representations from Transformers) for generating embeddings of bits of code or text. BERT is a pre-trained 
model trained with large amounts of text and textfiles.

It uses the encoder role of the Transformer Architecture.
    
In this context BERT tokenizer initialises the text as tokens with IDs that can be understood by the model. 

The tokens are then fed to the BERT model which produces contextual embeddings.

These embeddings are converted before feeding it to the Clustering algorithm as embedding is a vector which vary in length based on the content.To cluster effectively we need all the vectors to be the same size and hence we convert and pad the embeddings.

Afterwards,the cosine function is used to check how similar the embeddings are to each other based on their direction irrespective of their magnitude.

The cosine function involves normalization which involves dividing the dot product by the product of the magnitudes of the vectors.This ensures that the measure ranges from -1 to 1.

These values indicate:

1 indicates that the vectors are identical (i.e., the angle between them is 0 degrees).

0 indicates that the vectors are unrealated (i.e., the angle between them is 90 degrees).

-1 indicates that the vectors are diametrically opposed (i.e., the angle between them is 180 degrees).







2.3 Handling Large Code Files:
-

Large Code Files are handeled by breaking down the file into smaller files and using BERT tokenization.

BERT uses Named Entity Recognition (NER) for indentifying and classifying entities like functions,classes and variables.It also can extract relevant information like comments and strings.

It is broken down further by semantic matching which is then embedded and later clustered based on similarity. 

Clustering helps in grouping similar code sections based on functionality and it also identifies common features and patterns in large datasets.
Large code files are analysed and grouped based on similarity while filtering unnecessary content which is then used for documentation generation.

GPT then is used to generate content for the documentation as well adds some additional features like code refactoring which helps organsise such large code files.
GPT simply uses prompt engineering and simplifies the way we work with such large code files.






2.4 Maintaning Context In Agglomerative Clustering:
-

Agglomerative Clustering is a hierarchical clustering method that starts with each sample as its own cluster and merges clusters based on a distance 
metric until all points are in a cluster or until n clusters are created.

Here clustering starts off with each data point or in this case each embedding in the list as a cluster.Then the clusters are merged based on similarity 
determined by cosine similarity method previously mentioned.

Before clustering BERT embeddings which uses bidirectional context and cosine similarity are applied to maintain context.After clustering the output is evaluated to ensure that the context is maintained.
    
2.5 Efficient Documentation Generation:
-
After clustering,each cluster generates documentation and API Integration is used to send prompts and receive generated text.
Using GPT-3.5.It stands for Generative Pre-trained Transformer.It is a pre-trained model which generates human-like-text based on some prompt.It also uses Transformer Architecture.

In this case the document is generated based on some user prompt which specifies the structure and components of the documentation to be generated.Before this the codebase is splits code and uses parallel processing to process chunks of the code which is then fed to the GPT model.

The user prompt is specific prompt and then the system prompt is also mentioned which is the prompt responsible for context to help GPT to generate the needed output.

Each cluster generates documentation and API Integration is used to send prompts and receive generated text.
With the help of GPT along with documentation tests are generated which is a useful addition to efficient documentation. 







3.-The Tasks Solved Are:
-

3.1 Addition of Dendrogram Feature:
-


-A dendrogram is a diagram that shows the hierarchical relationship between objects.It is generated after clustering. The main use of a dendrogram is to work out the best way to allocate objects to clusters.The key to understanding a dendrogram is to notice which link height is the lowest between two clusters which suggests the most similarity between them.


-In this case dendrogram can be generated by including matplotlib to graph the converted embeddings and clustered elements obtained in the form of a list.The dendrogram has the x and y axis and the fig size can be manipulated as needed.

CODE


    Z = hierarchy.linkage(list1, method='complete', metric='cosine') 

    plt.figure(figsize=(10, 6))
    
    hierarchy.dendrogram(Z)
    
    plt.title('Dendrogram')
    
    plt.xlabel('Data points')
    
    plt.ylabel('Distance')
    
    plt.show()
    




-The plt.show() shows the dendrogram and if the dendogram is saved using plt.save(image/path) it can be displayed in the final generated documentation to provide the user with a graphical representation. 

The code is written in convert_embeddings.py and the route is added in routes.py.


3.2 Documentation Customization:
-

-The documentation generation uses GPT-3.5.It stands for Generative Pre-trained Transformer (GPT), a family of LLMs created by OpenAI that uses deep learning to generate human-like, conversational text.

-It generates meaningful text based on provided prompts and input text. This involves using machine learning.As of now the prompt is fixed in the code written for generating documentation.


-So, to be able to customize the generated documentation based on user we edit the code such that the prompt is taken from the user and each word is stored in a list which is then fed to the GPT model.This generates the required documentation.


    def call_openai_api_higher_tokens(text, output_file):

      def generate_text(messages):
    
         response = openai.ChatCompletion.create(
        
            model="gpt-3.5-turbo-16k",
            
            messages=messages,
            
            max_tokens=2000,
            
            n=1,
            
            stop=None,
            
            temperature=0.5,
            
        )
        
        return response.choices[0].message['content']
        

    messages = [
    
        {"role": "system", "content": "You are a smart technical writer who understands code and can write documentation for it."},
        
        {"role": "user", "content": f"Give me a developers documentation of the following code. Give a brief intro, table of contents, function explanations,
        dependencies, API specs (if present), schema tables in markdown. Give in markdown format and try to strict to the headings\n\n: {text}."},
    ]

    while True:
    
        response = generate_text(messages)
        
        print(response)
        

        feedback = input("Is there any more customisation you would like to add? (yes/no): ")
        
        if feedback.lower() == "yes":
        
            break
            
        else:
        
            refinement = input("How can I customise your documentation? ")
            
            messages[1]["content"] += " " + refinement

    save_to_file(response, output_file)

    This code is written in infinite_gpt.py.



3.3 Additional Code Features
-

-The additional code feature I added similar to code refactoring or test generation is code optimization.Code Optimization uses the same format of code as code refactoring and test generation.

-It also uses the GPT-3.5 model whcih takes the prompt to optimize the code and the model generates the feature which involves suggestions on how to optimize the code in the generated documentation.

    def ask_gpt_to_optimize_code(prompt_text, output_folder):
   
      system_prompt = """You are a skilled software engineer specializing in code optimization and performance improvements."""

      user_prompt = """Analyze the following code for performance bottlenecks and suggest optimizations to improve its efficiency. Provide only the optimized code 
      and avoid any additional explanations or comments."""
    
    output_file = f'{output_folder}/optimized_code.txt'

    print(SHOULD_MOCK_AI_RESPONSE)

    if SHOULD_MOCK_AI_RESPONSE == 'True':
    
        print("Mocking AI response")
        
        mock_chunks_gpt(prompt_text, output_file)
        
    elif SHOULD_MOCK_AI_RESPONSE == 'False':
    
        print("Calling OpenAI API")
        
        response = call_openai_api(prompt_text, system_prompt, user_prompt)
        
        print(response)
        
        save_to_file(response, output_file)


-Code Optimization can be a useful feature for users who need input on what can be further improved in their codebase to increase efficiency and better it.

There are many other additional features that can be added as GPT has the ability to generate a lot of useful stuff with the appropriate prompt.Other methods are giving code recommendations or changing code to different languages.

This code is written in infinite_gpt.py and the necessary route in routes.py.


3.4 Investigating Clustering
-

-Another alternative type of clustering that I thought would be a good fit was DBSCAN clustering.
Hierarchical clustering workS for finding spherical-shaped clusters or convex clusters. In other words, they are suitable only for compact and well-separated clusters.

Real-life data may contain irregularities, like:

Clusters can be of arbitrary shape such as those shown in the figure below. 

Data may contain noise.

-So,this can be overcome by DBSCAN.It stands for Density-Based Spatial Clustering Of Applications With Noise.It overcomes these limitations.

The steps involved are:

-Find all the neighbor points within eps and identify the core points or visited with more than MinPts neighbors.

(eps: It defines the neighborhood around a data point i.e. if the distance between two points is lower or equal to ‘eps’ then they are considered neighbors. If the eps value is chosen too small then a large part of the data will be considered as an outlier. If it is chosen very large then the clusters will merge and the majority of the data points will be in the same clusters. One way to find the eps value is based on the k-distance graph.

MinPts: Minimum number of neighbors (data points) within eps radius. The larger the dataset, the larger value of MinPts must be chosen. As a general rule, the minimum MinPts can be derived from the number of dimensions D in the dataset as, MinPts >= D+1.)
 

-For each core point if it is not already assigned to a cluster, create a new cluster.

-Find recursively all its density-connected points and assign them to the same cluster as the core point. 

(A point a and b are said to be density connected if there exists a point c which has a sufficient number of points in its neighbors and both points a and b are within the eps distance. This is a chaining process. So, if b is a neighbor of c, c is a neighbor of d, and d is a neighbor of e, which in turn is  neighbor of a implying that b is a neighbor of a.)

-Iterate through the remaining unvisited points in the dataset. Those points that do not belong to any cluster are noise.

CODE

    from sklearn.cluster import DBSCAN
  
    from sklearn.metrics import silhouette_score
  
    import numpy as np
  

    def tune_dbscan(X):
 
     best_score = -1
  
     best_params = []
  
     eps_range=np.arange(10,50,5)
  
     min_samples_range=range(10,50)
  
     for eps in eps_range:
  
       for min_samples in min_samples_range :
    
         clustering = DBSCAN(eps=eps, min_samples=min_samples)
      
         labels = clustering.fit_predict(X)
      
         if len(set(labels)) > 1: 
      
          score = silhouette_score(X, labels)
        
          if score > best_score:
        
            best_score = score
          
            best_params = [eps,min_samples]
          
       return best_params


    def DBclustering(list1):
  
     X = np.array(list1)
    
     list1=[]
    
     list1=tune_dbscan(X)
    
     e=list1[0]
    
     m=list1[1]

     dbscan = DBSCAN(eps=e, min_samples=m, metric='cosine').fit(X)
    
     arr = dbscan.labels_
    
     unique_values = np.unique(arr)
    
     indices_list = []
    
     for val in unique_values:
    
        indices = np.where(arr == val)[0]
        
        indices_list.append(indices)
    
     return indices_list


-The code involves tune_dbscan function which is responsible for finding the optimal eps and mins values(hypertuning)

-The obtained values are then used in the DBSCAN clustering algorithm which performs the clustering whcih then can be used for other processes along the lines like document generation in our case.

There are many other clustering methods like K means clustering,Mean shift clustering,spectral clustering etc.
The reason I choose DBSCAN is because I felt it was the best for handling large datasets, datasets of varying density and noise.

This code is written in seperate file dbscanclustering.py and is added in the similar format of the current clustering used.

3.5 Investigating Embeddings Mechanisms:
-


RoBERTa (Robustly optimized BERT approach) is a variant of BERT (Bidirectional Encoder Representations from Transformers).It builds on the BERT architecture but incorporates several improvements to enhance its performance. 

RoBERTa is trained on a larger datasets as compared to BERT.RoBERTa uses dynamic masking as opposed to static masking.RoBERTa focuses on the Masked Language Model (MLM).


    from transformers import RobertaTokenizer, RobertaModel
  
    import torch
  
    import numpy as np
  

    model_name = 'roberta-base'
   
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
   
    model = RobertaModel.from_pretrained(model_name)

    def code(codes):
  
     code_tokens = tokenizer.encode(codes, add_special_tokens=True)
    
     tokens_ids = torch.tensor(code_tokens)[None, :]
    
     with torch.no_grad():
    
        context_embeddings = model(tokens_ids)[0]
    
     return context_embeddings

    def embedding(text):
  
     t = 0
    
     window = 500
    
     overlap = 200
    
     step = 0
    
     size = len(text)
    
     b = np.zeros((1, 768)) 
    
     while (t < size):
    
        chunk = text[t:t+window]
        
        
        chunk_embeddings = code(chunk).detach().numpy().mean(axis=1)
        
        size_a = chunk_embeddings.size
        
        size_b = b.size
        
        diff = size_b - size_a
        
        if diff > 0:
        
            chunk_embeddings = np.pad(chunk_embeddings, (0, diff), 'constant')
            
        else:
        
            b = np.pad(b, (0, abs(diff)), 'constant')\
            
        b += chunk_embeddings
        
        t = t + window - overlap
        
        step += 1
        
        if (t >= size):
        
            t = size
            
     b = b / step
    
     b = b.reshape(int(b.size / 768), 768)
    
     return b

  
The cosine similarity code remains same as for BERT.
The above code is written in createembeddingsro.py.

 
    def convert_embeddings(text_list):
  
    reshaped_embeddings_list = []
    
    for text in text_list:

        embeddings = code(text).detach().numpy()
    
        mean_embedding = embeddings.mean(axis=1)
        
        reshaped_embedding = mean_embedding.reshape(-1)
        
        reshaped_embeddings_list.append(reshaped_embedding)
    
      return reshaped_embeddings_list

This code is written in convertembeddings.py.

Also the necessary additions are made in process.py file.

roBERTa just seems to be a better version of BERT.There are many other embedding models like ELMo for example but BERT seems to be the best model in terms of dealing wiht contextual embeddings.




4.Any Limitations or Improvements:
-
Some limitations are:

-Integrating APIs correctly requires up-to-date knowledge of their usage, which the model might not always possess.
So,it is necessary to constantly be up to date.

-BERT is currently used but I think a better version of itself which is RoBERTa can be used.

Other than the above I feel the codebase is well programmed and it is a project that will definitely help and can be utilised to its full potential.








    
