import re
from typing import Dict, List, Set, Tuple


class SimpleRAGPipeline:
    """
    Source-grounded RAG pipeline used for AI testing and evaluation.

    The pipeline is intentionally extractive so that every generated answer can
    be checked against retrieved evidence. This makes the project suitable for
    RAG regression testing, hallucination-risk analysis, and LLM evaluation
    portfolio demonstrations.
    """

    BROAD_DOMAIN_TERMS = {
        "usually", "customer", "customers", "product", "products", "company",
        "provide", "support", "account", "payment", "shipping", "return",
        "returns", "warranty", "order", "orders", "method", "methods", "item",
        "items", "policy", "business", "days", "take", "takes"
    }

    HIGH_SIGNAL_ABSENT_TERMS = {
        "cryptocurrency", "crypto", "bitcoin", "international", "overseas",
        "paypal", "venmo", "alipay", "wechat", "klarna", "cash", "cod"
    }

    CHINESE_HIGH_SIGNAL_ABSENT_TERMS = {
        "加密货币", "比特币", "国际配送", "国际运输", "海外配送", "海外运输",
        "支付宝", "微信支付", "货到付款"
    }

    def __init__(
        self,
        vector_store,
        top_k: int = 3,
        min_similarity_score: float = 0.05,
        min_question_coverage: float = 0.08,
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_similarity_score = min_similarity_score
        self.min_question_coverage = min_question_coverage

    def retrieve(self, question: str) -> List[Dict[str, object]]:
        """Retrieve relevant chunks from the vector store."""
        return self.vector_store.search(question, top_k=self.top_k)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Simple sentence splitter."""
        sentences = re.split(r"(?<=[.!?。！？])\s*", text.strip())
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        ascii_tokens = re.findall(r"[a-zA-Z0-9$]+", text.lower())
        chinese_tokens = []
        for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            chinese_tokens.append(sequence)
            for size in (2, 3):
                chinese_tokens.extend(
                    sequence[i:i + size]
                    for i in range(0, max(len(sequence) - size + 1, 0))
                )
        return ascii_tokens + chinese_tokens

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    @classmethod
    def _important_words(cls, text: str) -> Set[str]:
        """Extract simple keywords from a question or evidence sentence."""
        words = cls._tokenize(text)
        stopwords = {
            "what", "when", "where", "which", "who", "how", "does", "do",
            "is", "are", "can", "could", "should", "the", "a", "an",
            "to", "for", "of", "in", "on", "with", "and", "or", "if",
            "it", "its", "their", "my", "your", "our", "they", "them", "be",
            "by", "after", "before", "from", "as", "using", "use", "about"
        }
        return {word for word in words if word not in stopwords and len(word) > 2}

    @classmethod
    def _normalize_token(cls, token: str) -> str:
        """Very small normalization helper for rule-based matching."""
        token = token.lower().strip()
        if token.endswith("ied") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("ed") and len(token) > 4:
            return token[:-2]
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("es") and len(token) > 4:
            return token[:-2]
        if token.endswith("s") and len(token) > 3:
            return token[:-1]
        return token

    @classmethod
    def _normalized_word_set(cls, text: str) -> Set[str]:
        return {cls._normalize_token(token) for token in cls._important_words(text)}

    @staticmethod
    def _numeric_terms(text: str) -> Set[str]:
        return set(re.findall(r"\b\d+(?:\.\d+)?\b|\$\d+(?:\.\d+)?", text.lower()))

    @staticmethod
    def _day_constraints(text: str) -> Set[str]:
        english_days = re.findall(r"\b\d+(?:\.\d+)?\s+(?:business\s+)?days\b", text.lower())
        chinese_days = re.findall(r"\d+(?:\.\d+)?\s*(?:个)?(?:工作日|天|日)", text)
        return set(english_days + chinese_days)

    @classmethod
    def _context_text(cls, retrieved_chunks: List[Dict[str, object]]) -> str:
        return "\n".join(str(chunk.get("text", "")) for chunk in retrieved_chunks)

    @classmethod
    def _missing_context_terms(
        cls,
        question: str,
        retrieved_chunks: List[Dict[str, object]],
    ) -> Tuple[Set[str], float]:
        question_terms = {cls._normalize_token(term) for term in cls._important_words(question)}
        context_terms = cls._normalized_word_set(cls._context_text(retrieved_chunks))
        if not question_terms:
            return set(), 1.0
        covered_terms = question_terms.intersection(context_terms)
        coverage = len(covered_terms) / len(question_terms)
        missing_terms = question_terms - context_terms
        return missing_terms, coverage

    def _should_refuse_as_unanswerable(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, object]],
    ) -> Tuple[bool, str]:
        """
        Decide whether the retrieved evidence is insufficient for a safe answer.

        This guard is deliberately conservative for RAG evaluation. It prevents
        the system from answering questions that contain specific entities,
        payment methods, countries/shipping scope, or numeric constraints that
        are absent from the retrieved context.
        """
        if not retrieved_chunks:
            return True, "no retrieved evidence"

        context = self._context_text(retrieved_chunks).lower()
        question_lower = question.lower()
        missing_terms, coverage = self._missing_context_terms(question, retrieved_chunks)
        is_chinese_question = self._contains_chinese(question)
        question_days = sorted(self._day_constraints(question_lower))

        if is_chinese_question and "退货" in question and question_days:
            missing_day_constraints = [day for day in question_days if day not in context]
            if missing_day_constraints:
                mentioned_days = sorted(self._day_constraints(context))
                mentioned_text = f" 文档只提到{mentioned_days[0]}。" if mentioned_days else ""
                return (
                    True,
                    f"提供的文档没有提供{missing_day_constraints[0]}后退货的依据。"
                    f"{mentioned_text}我无法确认{missing_day_constraints[0]}后可以退货",
                )

        for term in self.CHINESE_HIGH_SIGNAL_ABSENT_TERMS:
            if term in question and term not in context:
                if term in {"国际配送", "国际运输", "海外配送", "海外运输"}:
                    return True, "提供的文档没有提到国际配送"
                if term in {"加密货币", "比特币"}:
                    return True, "提供的文档没有提到加密货币作为可接受的付款方式"
                if term in {"支付宝", "微信支付"}:
                    return True, f"提供的文档没有提到{term}作为可接受的付款方式"
                return True, f"提供的文档没有提到{term}"

        # Missing numeric constraints are high risk. Example: "after 90 days"
        # should not be answered using a generic 30-day return sentence.
        question_numbers = self._numeric_terms(question_lower)
        context_numbers = self._numeric_terms(context)
        missing_numbers = question_numbers - context_numbers
        if missing_numbers:
            missing_number_text = ", ".join(sorted(missing_numbers))
            mentioned_days = sorted(self._day_constraints(context))
            asks_about_return = "return" in question_lower or "退货" in question
            if asks_about_return and question_days:
                mentioned_text = f" The documents only mention {mentioned_days[0]}." if mentioned_days else ""
                if is_chinese_question:
                    mentioned_text = f" 文档只提到{mentioned_days[0]}。" if mentioned_days else ""
                    return (
                        True,
                        f"提供的文档没有提供{question_days[0]}后退货的依据。"
                        f"{mentioned_text}我无法确认{question_days[0]}后可以退货",
                    )
                return (
                    True,
                    "the provided documents do not provide support for returning products "
                    f"after {question_days[0]}.{mentioned_text} I cannot confirm returns "
                    f"after {question_days[0]}",
                )
            return True, f"numeric constraint not found in evidence: {missing_number_text}"

        high_signal_missing = {
            term for term in missing_terms
            if term in self.HIGH_SIGNAL_ABSENT_TERMS
        }
        if high_signal_missing:
            if "international" in high_signal_missing and "shipping" in question_lower:
                return True, "the provided documents do not mention international shipping"
            if {"cryptocurrency", "crypto", "bitcoin"}.intersection(high_signal_missing):
                return (
                    True,
                    "the provided documents do not mention cryptocurrency as an accepted "
                    "payment method",
                )
            return True, f"specific term not found in evidence: {', '.join(sorted(high_signal_missing))}"

        specific_missing_terms = {
            term for term in missing_terms
            if len(term) >= 8 and term not in self.BROAD_DOMAIN_TERMS
        }
        yes_no_question = bool(re.match(r"\s*(can|does|do|is|are|will|would|could|should)\b", question_lower))
        yes_no_question = yes_no_question or bool(re.match(r"\s*(可以|能否|是否|会不会|能不能)", question))
        if yes_no_question and specific_missing_terms:
            return True, f"specific question term not found in evidence: {', '.join(sorted(specific_missing_terms))}"

        if coverage < self.min_question_coverage:
            return True, f"low question-evidence coverage: {coverage:.2f}"

        return False, "evidence appears sufficient"

    @staticmethod
    def _safe_no_answer(reason: str) -> str:
        return (
            "I could not find enough information in the provided documents to answer this safely. "
            f"Reason: {reason}."
        )

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, object]]
    ) -> str:
        """
        Generate a simple grounded answer from retrieved chunks.

        The answer is extractive and contains a source citation. If the evidence
        is not sufficient, the method returns a safe refusal instead of guessing.
        """
        if not retrieved_chunks:
            return self._safe_no_answer("no retrieved evidence")

        should_refuse, reason = self._should_refuse_as_unanswerable(question, retrieved_chunks)
        if should_refuse:
            return self._safe_no_answer(reason)

        best_score = retrieved_chunks[0].get("similarity_score", 0)
        if best_score < self.min_similarity_score:
            return self._safe_no_answer("retrieval similarity below threshold")

        question_words = self._normalized_word_set(question)
        candidate_sentences = []

        for chunk in retrieved_chunks:
            source = chunk["source"]
            for sentence in self._split_sentences(chunk["text"]):
                sentence_words = self._normalized_word_set(sentence)
                overlap = len(question_words.intersection(sentence_words))
                candidate_sentences.append({
                    "sentence": sentence,
                    "source": source,
                    "overlap": overlap,
                    "similarity_score": chunk.get("similarity_score", 0)
                })

        if not candidate_sentences:
            return self._safe_no_answer("no usable evidence sentences")

        candidate_sentences = sorted(
            candidate_sentences,
            key=lambda x: (x["overlap"], x["similarity_score"]),
            reverse=True
        )

        useful_sentences = [item for item in candidate_sentences if item["overlap"] > 0]
        selected = useful_sentences[:2] if useful_sentences else candidate_sentences[:1]

        answer_parts = [item["sentence"] for item in selected]
        sources = sorted({item["source"] for item in selected})
        return " ".join(answer_parts) + f" Source: {', '.join(sources)}."

    def ask(self, question: str) -> Dict[str, object]:
        """Run retrieval and answer generation."""
        retrieved_chunks = self.retrieve(question)
        answer = self.generate_answer(question, retrieved_chunks)
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "retrieved_context": self._context_text(retrieved_chunks),
            "sources": [chunk["source"] for chunk in retrieved_chunks]
        }
