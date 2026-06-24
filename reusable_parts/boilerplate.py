"""
boilerplate.py
--------------
Shared prompt template + Pydantic schema for thesis-abstract requirement
extraction. Extracted from program.ipynb cells 1 and 3 so app.py (and any
other script) can import them without duplicating the text.
"""

from pydantic import BaseModel
from typing import List
from enum import Enum

class DegreeLevel(str, Enum):
    BSc = "BSc"
    MSc = "MSc"
    PhD = "PhD"
    MSc_to_PhD = "MSc-to-PhD"

class Category(str, Enum):
    data = "data"
    method_technique = "method_technique"
    tool_library = "tool_library"
    compute = "compute"
    human_effort = "human_effort"
    other = "other"

class Difficulty(str, Enum):
    basic = "basic"
    intermediate = "intermediate"
    advanced = "advanced"

class Requirement(BaseModel):
    category: Category
    name: str
    description: str
    estimated_difficulty: Difficulty
    search_query: str

class AnalysisResult(BaseModel):
    overall_suitable_degree: DegreeLevel
    requirements: List[Requirement]



prompt = """You are an experienced Computer Science professor and MSc/PhD supervisor with 20+ years of experience. 
Your task is to carefully analyze a thesis abstract and extract the key **resource requirements** needed to carry out the described research.

Be strict and conservative. Do NOT hallucinate or invent resources that are not supported by the text. If the abstract is from a non-Computer Science field, still extract meaningfully but do not force CS-style tools.
Try to be as verbatim as possible in extracting the requirements, and do not infer beyond what is supported by the text. If a requirement is implied but not explicitly stated, still include it, but do not add any requirements that are not supported by the text.

---

### Example

**Abstract:**
"This thesis is an examination of the national narratives contained in three exhibits in The Musem of New Zealand, Te Papa Tongarewa. It examines the existence of the state and the nation, and their involvement in museum development, and applies this theory, and selected theories of Roland Barthes, Sergei Eisenstein, and Walter Benjamin, to the subsequent analysis. Broadly, the position taken is that museums are one of a number of institutions that perpetuate national narratives in order to bind nations together and discourage anti-state sentiment, and this position is validated in the analysis of three long-term Te Papa exhibits, Exhibiting Ourselves, Parade, and Golden Days."

**Expected Output:**
```json
{{
  "overall_suitable_degree": "MSc",
  "requirements": [
    {{
      "category": "method_technique",
      "name": "National narrative analysis",
      "description": "Examination of national narratives in museum exhibits",
      "estimated_difficulty": "intermediate",
      "suitable_degree": "MSc"
    }},
    {{
      "category": "method_technique",
      "name": "Application of cultural theory",
      "description": "Application of theories from Roland Barthes, Sergei Eisenstein, and Walter Benjamin",
      "estimated_difficulty": "advanced",
      "suitable_degree": "PhD"
    }},
    {{
      "category": "data",
      "name": "Te Papa Tongarewa exhibits",
      "description": "Analysis of three long-term exhibits: Exhibiting Ourselves, Parade, and Golden Days",
      "estimated_difficulty": "intermediate",
      "suitable_degree": "MSc"
    }},
    {{
      "category": "method_technique",
      "name": "Institutional analysis",
      "description": "Analysis of the role of the state and nation in museum development",
      "estimated_difficulty": "intermediate",
      "suitable_degree": "MSc"
    }}
  ]
}}

---

Now apply the same analysis to the following abstract:

[THESIS ABSTRACT]
{abstract}
[/THESIS ABSTRACT]

Focus on what would actually be required in terms of:
- Datasets / data sources
- Methods, algorithms, or theoretical techniques
- Tools, libraries, or frameworks
- Compute / hardware resources
- Human effort / skills
- Any other critical resources

### Field generation rules by category

**For `method_technique` and `tool_library`:**
Generate an `search_query` field.
- Use technical terms, acronyms, and key context words from the abstract.
- Use short, precise phrases of 3–4 words to capture the core of the technical or methodological requirement.
- Do not use author names in these queries; focus on the method or tool itself.

**For `data`:**
generate a `search_query` field that is
intended to locate THIS specific dataset or corpus if it exists
  publicly. Use the most identifying information available in the abstract —
  typically the dataset name, the citing author if named, or the institution.
  Template: [dataset_name or corpus_identifier] [domain] [language]


### Output Format
Return a valid JSON object with exactly this structure:

```json
{{
  "overall_suitable_degree": "BSc" | "MSc" | "PhD" | "MSc-to-PhD",
  "requirements": [
    {{
      "category": "data" | "method_technique" | "tool_library" | "compute" | "human_effort" | "other",
      "name": "Short clear name of the resource",
      "description": "One sentence explaining how it is used in the work",
      "estimated_difficulty": "basic" | "intermediate" | "advanced",
      "search_query": "Optimized search query for this requirement"
    }}
  ]
}}
```

Return only valid JSON. Do not include any explanation, markdown fencing, or preamble outside the JSON object.
    """
