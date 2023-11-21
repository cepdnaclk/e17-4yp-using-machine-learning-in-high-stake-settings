
var documents = [{
    "id": 0,
    "url": "//404.html",
    "title": "404",
    "body": "404 Page does not exist!Please use the search bar at the top or visit our homepage! "
    }, {
    "id": 1,
    "url": "//about",
    "title": "Team TeaPot and our Work",
    "body": "            Madhushan   e17194@eng. pdn. ac. lk   University of Peradeniya            Ravisha   e17296@eng. pdn. ac. lk   University of Peradeniya            Thanujan   e17342@eng. pdn. ac. lk   University of Peradeniya            Dr. Damayanthi   damayanthiherath@eng. pdn. ac. lk   University of Peradeniya            Dr. Kasun   amarasinghek@cmu. edu   Carnegie Mellon University            Dr. Upul   upuljm@eng. pdn. ac. lk   University of Peradeniya    Welcome to Teapot, a dynamic research team comprising three dedicated computer engineering undergraduates and led by two esteemed computer engineering lecturers from the University of Peradeniya. In addition, we are fortunate to have the guidance of a seasoned Senior Research Scientist from Carnegie Mellon University. Our MissionAt Teapot, our mission is to delve into the cutting-edge domain of the disagreement problem and explore the latest evaluation metrics introduced in the field. In 2023, we embarked on a research journey to unravel the intricacies of Human-Machine Learning (ML) interaction using Explainable Artificial Intelligence (XAI). Research FocusTeapot is dedicated to advancing the understanding of the disagreement problem and its implications in the realm of Human-ML interaction. Our research centers around exploring the barriers inherent in this interaction through the lens of Explainable AI, utilizing the most recent methodologies and insights available. Join us on this exciting journey as we strive to push the boundaries of knowledge and contribute to the ever-evolving landscape of artificial intelligence. Research Gap Identified in the FYP   Our Plan to Bridge the Research Gap   Dataset We ChoseOur study relies on the DonorsChoose dataset (2014-2018), encompassing Projects. csv, Donations. csv, Schools. csv, and Teachers. csv. We've amalgamated these datasets, yielding 1,110,015 unique projects. Our focus extends to teacher donations, particularly analyzing four months post-project posting. We calculate total donations, determining the percentage of the project cost covered. Currently, we haven't explored converting text columns to NLP features. This dataset exploration forms the foundation for our research, providing insights into project funding dynamics and community engagement within the realm of Explainable AI. To get access to the dataset Contact Us Contact Donors ChooseOur Prediction Pipeline   XAI Pipeline We Developed   Evaluation of Disagreement   Experiments for analysis   Questions?: Head over to our Github repository!or Write to Us. . .  "
    }, {
    "id": 2,
    "url": "//categories",
    "title": "Categories",
    "body": ""
    }, {
    "id": 3,
    "url": "//",
    "title": "Home",
    "body": "      Featured:                                                                                                                                                                                                                 Navigating the Complexity: A Deep Dive into Explainable AI                              :               Welcome back to the intriguing world of Artificial Intelligence, where today, we’re set to explore the nuances of Explainable AI (XAI) specifically tailored for engineering. . . :                                                                                                                                                                       Madhushan                                06 Oct 2023                                                                                                                                                                                                                                                                                                                  An Introduction to Explainable AI                              :               Have you ever wondered how your smartphone magically understands your voice commands or how a computer predicts what movies you might like? Enter the world. . . :                                                                                                                                                                       Madhushan                                20 Sep 2023                                                                                                                      All Stories:                                                                                                     Unveiling the Challenge of Disagreements: A summary              :       In the expansive landscape of artificial intelligence, the quest for transparency and interpretability has given rise to a myriad of tools collectively known as explainable machine learning (XAI). In this. . . :                                                                               Ravisha                12 Oct 2023                                                                                                                                     Navigating the Complexity: A Deep Dive into Explainable AI              :       Welcome back to the intriguing world of Artificial Intelligence, where today, we’re set to explore the nuances of Explainable AI (XAI) specifically tailored for engineering undergrads. Let’s unravel the intricacies. . . :                                                                               Madhushan                06 Oct 2023                                                                                                                                     An Introduction to Explainable AI              :       Have you ever wondered how your smartphone magically understands your voice commands or how a computer predicts what movies you might like? Enter the world of Artificial Intelligence (AI) –. . . :                                                                               Madhushan                20 Sep 2023                                            "
    }, {
    "id": 4,
    "url": "//robots.txt",
    "title": "",
    "body": "      Sitemap: {{ “sitemap. xml”   absolute_url }}   "
    }, {
    "id": 5,
    "url": "//a-summary-of-quantifying-disagreement-problem/",
    "title": "Unveiling the Challenge of Disagreements: A summary",
    "body": "2023/10/12 - In the expansive landscape of artificial intelligence, the quest for transparency and interpretability has given rise to a myriad of tools collectively known as explainable machine learning (XAI). In this context, a critical issue has emerged—disagreements among these explanations. This blog delves into the intricacies of the disagreement problem, exploring the difficulties faced by practitioners and suggesting potential paths for resolution summarized from the research held by Satyapriya Krishna, Tessa Han, Alex Gu, Javin Pombra, Shahin Jabbari, Zhiwei Steven Wu, and Himabindu Lakkaraju. This research is a collaborative work of the world’s renowned institutes Harvard University, Massachusetts Institute of Technology, Drexel University, and Carnegie Mellon University The Disagreement Problem: Disagreements among various XAI tools have become a common hurdle in real-world applications, posing a threat to the accuracy and reliability of ML models. This issue becomes particularly acute in critical domains where ML models are deployed. Unfortunately, the absence of a standardized methodology for resolving these disagreements compounds the complexity, making it challenging for practitioners to confidently rely on the decisions made by ML models. Background: Understanding the XAI Toolbox: To comprehend the disagreement problem, we must first navigate through the two main categories of XAI methods: inherently interpretable models and post hoc explanations. Inherently interpretable models, like Generalized Additive Models (GAMs) and decision trees, offer simplicity but come with a trade-off in model complexity. This trade-off has led to the prevalence of post hoc explanation methods, including popular techniques such as LIME, SHAP, and various gradient-based approaches. Previous studies have attempted to evaluate the fidelity and stability of these explanations, introducing metrics such as fidelity, stability, consistency, and sparsity. However, as research progressed, the discovery of inconsistencies and vulnerabilities within existing explanation methods, including susceptibility to adversarial attacks, raised concerns about their reliability. Methodology: Unraveling Disagreements: This study, conducted by Krishna S. and researchers, addressed the disagreement problem through a multifaceted approach: Semi-Structured Interviews:Interviews with 25 data scientists revealed that 88% of practitioners utilize multiple explanation methods, with 84% encountering frequent instances of disagreement. Factors contributing to disagreement include different top features, ranking among top features, signs in feature contribution, and relative ordering among features. Framework for Quantifying Disagreement:The researchers designed a novel framework to quantitatively measure disagreement using six metrics: feature agreement, rank agreement, sign agreement, signed rank agreement, rank correlation, and pairwise rank agreement. These metrics provide a comprehensive evaluation of disagreement levels. Empirical Analysis:Employing four datasets, six popular explanation methods, and various ML models, the researchers conducted an empirical analysis that uncovered trends in disagreement based on model complexity and granularity of data representation (tabular, text, and image). Notably, disagreement tends to increase with model complexity. Qualitative Study:A qualitative study explored decisions made by data scientists when facing explanation disagreements. Findings revealed a lack of formal agreement on decision-making, with participants relying on personal heuristics and preferences for certain methods. Results: Illuminating the Path Forward: The results of this comprehensive study offer valuable insights: Frequency of Disagreement:The researchers observed a high occurrence of disagreement among explanation methods, prompting the need for a systematic approach to navigate these disparities. Heuristics and Preferences:ML practitioners often rely on personal heuristics and preferences when selecting explanation methods, highlighting the subjective nature of decision-making in the face of disagreement. Metrics for Quantifying Disagreement:The introduced framework with six quantitative metrics provides a robust means of assessing and comparing disagreement levels, enhancing our understanding of the complexities involved. Conclusion and Future Directions: In conclusion, the disagreement problem in XAI demands attention and strategic solutions. The study, conducted by Krishna S. and researchers, not only uncovers the prevalence of disagreement but also introduces a framework for its quantitative measurement. Future research should delve into the root causes of disagreement, propose innovative resolution methods, and establish reliable evaluation metrics. As we navigate the intricate landscape of XAI, the journey is marked by challenges, discoveries, and the collective effort of practitioners and researchers alike, seeking clarity in the face of disagreement. Regular education and awareness are crucial to equip data scientists with the latest approaches and foster a global community committed to advancing the field of explainable AI. "
    }, {
    "id": 6,
    "url": "//a-deep-dive-into-xai/",
    "title": "Navigating the Complexity: A Deep Dive into Explainable AI",
    "body": "2023/10/06 - Welcome back to the intriguing world of Artificial Intelligence, where today, we’re set to explore the nuances of Explainable AI (XAI) specifically tailored for engineering undergrads. Let’s unravel the intricacies of Inherently Interpretable Models, delve into the realm of Post Hoc Explanations, and meet the tools – Lime and Shap – that illuminate the path to understanding AI decision-making. Inherently Interpretable Models:: In the realm of AI, Inherently Interpretable Models are akin to the sought-after Rosetta Stone, translating complex machine learning algorithms into understandable language. These models are designed to provide transparency from the get-go, ensuring that the inner workings of the decision-making processes are comprehensible. For engineering minds, think of it as having access to the source code of an algorithm, allowing you to trace each decision back to its roots. This transparency is crucial for understanding and fine-tuning models, making Inherently Interpretable Models a valuable asset for engineers delving into the intricate world of AI. Post Hoc Explanations:: Moving on to Post Hoc Explanations, consider this as a debug mode for AI decisions. It’s like having a detailed log file that explains every step the model took to arrive at a particular decision. For engineering undergrads, this is akin to post-mortem analysis – a critical tool for understanding and improving system performance. Post Hoc Explanations provide a detailed breakdown of decisions after they’ve been made. Imagine having a log of the execution path of your code, but for AI decision pathways. It’s not just about the result; it’s about gaining insights into the decision-making process itself. Lime &amp; Shap:: Now, let’s meet Lime and Shap – the analytical tools engineered to bring clarity to AI decision landscapes. Lime specializes in providing localized explanations, much like debugging a specific section of code. It zooms in on precise decisions, making it invaluable for engineers keen on pinpointing and optimizing specific aspects of an AI model. Shap takes a holistic approach, offering a comprehensive view of how each variable contributes to the final decision. It’s like having a system profiler for your AI model, revealing the significance of each input feature. Shap transforms the abstract into the concrete, enabling engineers to make informed decisions about model behavior. Real-world Application:: Now, let’s ground these concepts in real-world applications. Imagine optimizing an AI system for a critical engineering task. Inherently Interpretable Models provide the foundational understanding needed for efficient model development. Post Hoc Explanations become your diagnostic tools, ensuring that every decision aligns with engineering principles. In a practical scenario, Lime and Shap act as your debugging and profiling tools, allowing you to analyze and optimize the AI model’s performance. This level of transparency is indispensable for engineering undergrads aiming to design AI systems with precision and reliability. As engineering undergraduates, your journey into AI involves not just creating powerful models but also ensuring they align with engineering principles. Inherently Interpretable Models and Post Hoc Explanations, facilitated by tools like Lime and Shap, empower you to navigate the complexities of AI, offering transparency and control in the development process. So, let’s equip ourselves with these analytical tools as we continue to engineer the future of AI. "
    }, {
    "id": 7,
    "url": "//an-introduction-to-explainable-ai/",
    "title": "An Introduction to Explainable AI",
    "body": "2023/09/20 - Have you ever wondered how your smartphone magically understands your voice commands or how a computer predicts what movies you might like? Enter the world of Artificial Intelligence (AI) – the brains behind the digital magic! But, just like a wizard revealing the secrets of a spell, let’s uncover the mystery behind a special kind of AI called “Explainable AI. ” Imagine you have a super-smart robot friend named Robo-Buddy. Robo-Buddy can do amazing things, like predicting the weather or suggesting which ice cream flavor you’ll love. But here’s the catch: Robo-Buddy doesn’t just work its magic in secret; it explains how it does it! Now, let’s dive into the basics. Explainable AI, or XAI for short, is like having a conversation with Robo-Buddy. It’s not enough for it to say, “Wear a jacket today. ” Instead, it tells you, “Wear a jacket because it’s going to rain, and I noticed you don’t like getting wet. ” See? No magic words – just clear explanations. In the AI world, models usually make decisions based on complex patterns they find in data. Imagine Robo-Buddy looking at your ice cream choices. Instead of saying, “I think you’ll like chocolate,” it might say, “I noticed you always smile when you have chocolate, so I’m suggesting it!” That’s the magic of Explainable AI – it shows its work. Now, why does this matter? Imagine you’re denied a loan by a computer. With Explainable AI, it won’t just say “no. ” It’ll say, “Your loan request was denied because your income is below the required amount. ” It’s like having a friendly chat with the decision-maker, not a mysterious figure. So, in this world of AI wonders, let’s celebrate the power of understanding. With Explainable AI, the magic isn’t hidden – it’s a conversation that makes technology feel like a helpful friend, always ready to explain its tricks! "
    }];

var idx = lunr(function () {
    this.ref('id')
    this.field('title')
    this.field('body')

    documents.forEach(function (doc) {
        this.add(doc)
    }, this)
});
function lunr_search(term) {
    document.getElementById('lunrsearchresults').innerHTML = '<ul></ul>';
    if(term) {
        document.getElementById('lunrsearchresults').innerHTML = "<p>Search results for '" + term + "'</p>" + document.getElementById('lunrsearchresults').innerHTML;
        //put results on the screen.
        var results = idx.search(term);
        if(results.length>0){
            //console.log(idx.search(term));
            //if results
            for (var i = 0; i < results.length; i++) {
                // more statements
                var ref = results[i]['ref'];
                var url = documents[ref]['url'];
                var title = documents[ref]['title'];
                var body = documents[ref]['body'].substring(0,160)+'...';
                document.querySelectorAll('#lunrsearchresults ul')[0].innerHTML = document.querySelectorAll('#lunrsearchresults ul')[0].innerHTML + "<li class='lunrsearchresult'><a href='" + url + "'><span class='title'>" + title + "</span><br /><span class='body'>"+ body +"</span><br /><span class='url'>"+ url +"</span></a></li>";
            }
        } else {
            document.querySelectorAll('#lunrsearchresults ul')[0].innerHTML = "<li class='lunrsearchresult'>No results found...</li>";
        }
    }
    return false;
}

function lunr_search(term) {
    $('#lunrsearchresults').show( 400 );
    $( "body" ).addClass( "modal-open" );
    
    document.getElementById('lunrsearchresults').innerHTML = '<div id="resultsmodal" class="modal fade show d-block"  tabindex="-1" role="dialog" aria-labelledby="resultsmodal"> <div class="modal-dialog shadow-lg" role="document"> <div class="modal-content"> <div class="modal-header" id="modtit"> <button type="button" class="close" id="btnx" data-dismiss="modal" aria-label="Close"> &times; </button> </div> <div class="modal-body"> <ul class="mb-0"> </ul>    </div> <div class="modal-footer"><button id="btnx" type="button" class="btn btn-danger btn-sm" data-dismiss="modal">Close</button></div></div> </div></div>';
    if(term) {
        document.getElementById('modtit').innerHTML = "<h5 class='modal-title'>Search results for '" + term + "'</h5>" + document.getElementById('modtit').innerHTML;
        //put results on the screen.
        var results = idx.search(term);
        if(results.length>0){
            //console.log(idx.search(term));
            //if results
            for (var i = 0; i < results.length; i++) {
                // more statements
                var ref = results[i]['ref'];
                var url = documents[ref]['url'];
                var title = documents[ref]['title'];
                var body = documents[ref]['body'].substring(0,160)+'...';
                document.querySelectorAll('#lunrsearchresults ul')[0].innerHTML = document.querySelectorAll('#lunrsearchresults ul')[0].innerHTML + "<li class='lunrsearchresult'><a href='" + url + "'><span class='title'>" + title + "</span><br /><small><span class='body'>"+ body +"</span><br /><span class='url'>"+ url +"</span></small></a></li>";
            }
        } else {
            document.querySelectorAll('#lunrsearchresults ul')[0].innerHTML = "<li class='lunrsearchresult'>Sorry, no results found. Close & try a different search!</li>";
        }
    }
    return false;
}
    
$(function() {
    $("#lunrsearchresults").on('click', '#btnx', function () {
        $('#lunrsearchresults').hide( 5 );
        $( "body" ).removeClass( "modal-open" );
    });
});