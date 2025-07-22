# Real Estate Listing Analyzer by Illia Shybanov for CAI 2300C

## Problem Statement
Homebuyers and agents are overwhelmed by lengthy, unstructured property listings and limited comparative context, leading to slower decision-making and missed opportunities in fast-moving markets like Miami. Our solution uses AI-driven NLP and market analytics to transform verbose listings into concise, actionable insights—saving users time and improving confidence in property selection.

## Stakeholder Map
| Stakeholder                   | Needs / Pain Points                                                         | Value Delivered by Analyzer                                       |
|-------------------------------|------------------------------------------------------------------------------|--------------------------------------------------------------------|
| **Homebuyers & Renters**      | Too many long descriptions; difficulty comparing features and prices         | Instant summary, feature tags, and price-vs-market badge           |
| **Real Estate Agents/Brokers**| Manual listing write-up and market analysis is time-consuming                | Auto-generated marketing copy; comparative market insights         |
| **Listing Platforms (Zillow, etc.)** | High bounce rates; poor user engagement                                  | Enhanced UX with bite-sized summaries, tag-based search filters    |
| **Property Sellers**          | Inconsistent quality of listing descriptions                                 | AI-polished, complete listing text; recommendations for missing info |
| **Internal Development Team** | Clear requirements, measurable goals, and defined tech stack                 | This document, SMART goals, and folder structure                   |

## SMART Goals
1. **S**ummarization Accuracy: Achieve an average ROUGE-1 F1 score ≥ 0.35 on a 100-listing test set by week 4.  
2. **M**arket Analysis: Display price vs. neighborhood median within ± 5% error for 90% of test listings by week 6.  
3. **A**doption: Pilot with 5 agents, each generating ≥ 20 summaries in the first month, and gather ≥ 80% positive feedback.  
4. **R**esponse Time: Return full analysis (summary, tags, price badge) in ≤ 3 seconds per listing on average.  
5. **T**hreshold: Deploy MVP (summaries + tag extraction) to a demo URL by August 10, 2025.

## Tools & Technology
- **Backend Framework**: FastAPI (Python)  
- **NLP / Summarization**: spaCy, Hugging Face Transformers (T5 or GPT-3.5 via API)  
- **Market Data**: Zillow API (ZWSID key), Redfin Data Center CSVs, Kaggle DB  
- **Vector Search**: FAISS or Pinecone for similarity embeddings (Sentence-Transformers)  
- **Database**: PostgreSQL for static listings; Redis for caching  
- **Frontend**: React with Tailwind CSS  
- **DevOps**: Docker, GitHub Actions CI/CD  
- **Project Management**: Microsoft Planner for tasks; GitHub Issues for backlog