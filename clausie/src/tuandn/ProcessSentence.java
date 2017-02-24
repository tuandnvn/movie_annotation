package tuandn;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import java.util.stream.Collector;
import java.util.stream.Collectors;

import org.apache.commons.lang3.StringUtils;

import de.mpii.clausie.ClausIE;
import de.mpii.clausie.Clause;
import de.mpii.clausie.Constituent;
import de.mpii.clausie.Constituent.Type;
import de.mpii.clausie.IndexedConstituent;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.trees.EnglishGrammaticalRelations;
import edu.stanford.nlp.trees.Tree;

public class ProcessSentence {
	ClausIE clausIE;

	public ProcessSentence(ClausIE clausIE) {
		this.clausIE = clausIE;
	}

	// Make some small change to the original sentence
	// Adding a subject "he" after "Then"
	// Change all SOMEONE to someone
	public String preprocess(String sentence) {
		sentence = sentence.replaceAll("SOMEONE", "someone");
		sentence = sentence.replaceAll(" then ", " then he ");
		sentence = sentence.replaceAll("Then ", "Then he ");
		sentence = sentence.replaceAll(",then ", ",then he ");
		return sentence;
	}

	public List<SentenceTuple> postprocess(List<SentenceTuple> sentenceTuples) {
		return sentenceTuples;
	}

	public List<SentenceTuple> process(String sentence, boolean verbose, boolean retry) {
		sentence = preprocess(sentence);
		List<SentenceTuple> tuples = new Vector<>();

		this.clausIE.parse(sentence);
		this.clausIE.detectClauses();

		if (verbose) {
			System.out.print("Dependency parse : ");
			System.out.println(clausIE.getDepTree().pennString().replaceAll("\n", "\n ").trim());
			System.out.print("Semantic graph : ");
			System.out.println(clausIE.getSemanticGraph().toFormattedString().replaceAll("\n", "\n ").trim());

			String sep = "";
			for (Clause clause : clausIE.getClauses()) {
				System.out.println(sep + clause.toString(clausIE.getOptions()));
				sep = " ";
			}
		}

		List<Clause> clauses = this.clausIE.getClauses();
		for (Clause clause : clauses) {
			List<Constituent> constituents = clause.getConstituents();

			Map<Type, Constituent> constituentMap = new HashMap<>();

			for (int index = 0; index < constituents.size(); index++) {
				Constituent constituent = constituents.get(index);
				if (!constituentMap.containsKey(constituent.getType())) {
					constituentMap.put(constituent.getType(), constituent);
				}

				if (constituent.getType() == Type.ADVERBIAL) {
					switch (clause.getFlag(index, clausIE.getOptions())) {
					case IGNORE:
						break;
					case REQUIRED:
						constituentMap.put(constituent.getType(), constituent);
						break;
					default:
						break;
					}
				}
			}

			switch (clause.getType()) {
			// The best case scenario is that I can change from
			// There are three glasses of water on the TV set.
			// to the form: three glasses of water are on the TV set.
			// there are people milling around -> People are smiling around
			// However there is not many EXISTENTIAL sentences in the dataset
			// 3 sentences in val set
			// Not so many in training set
			// So let's just leave that for now
			case EXISTENTIAL:
				break;
			case SV:
			case SVA: {
				SentenceTuple st = null;

				// There is still chance that directObject is not null
				String directObject = null;

				if (constituentMap.containsKey(Type.DOBJ)) {
					directObject = constituentMap.get(Type.DOBJ).rootString();
				}

				// This still need to improved, for example:
				// The woman next to the table stands up
				// -> The womand stands up next to the table
				// We could use the prepositional phrase embedded in subject to replace the real prepositional phrase
				if (!constituentMap.containsKey(Type.ADVERBIAL)) {
					st = new SentenceTuple(constituentMap.get(Type.SUBJECT).rootString(),
							getVerbform(constituentMap.get(Type.VERB)), directObject, null, null);
				} else if (constituentMap.get(Type.ADVERBIAL).getClass() == IndexedConstituent.class) {
					IndexedConstituent c = (IndexedConstituent) (constituentMap.get(Type.ADVERBIAL));
					if (c.isPrepPhrase()) {

						if (c.getDependent() != null) {
							st = new SentenceTuple(constituentMap.get(Type.SUBJECT).rootString(),
									getVerbform(constituentMap.get(Type.VERB)), directObject, c.rootString(),
									c.getDependent().originalText());
						} else {
							st = new SentenceTuple(constituentMap.get(Type.SUBJECT).rootString(),
									getVerbform(constituentMap.get(Type.VERB)), directObject, c.rootString(), null);
						}
					} else {
						st = new SentenceTuple(constituentMap.get(Type.SUBJECT).rootString(),
								getVerbform(constituentMap.get(Type.VERB)), directObject, null, null);
					}
				}

				if (st != null)
					tuples.add(st);

			}
				break;

			case SVO:
			case SVOA: {
				SentenceTuple st = null;

				String directObject = null;

				if (constituentMap.containsKey(Type.DOBJ)) {
					directObject = constituentMap.get(Type.DOBJ).rootString();
				}

				if (!constituentMap.containsKey(Type.ADVERBIAL)) {
					st = new SentenceTuple(constituentMap.get(Type.SUBJECT).rootString(),
							getVerbform(constituentMap.get(Type.VERB)), directObject, null, null);
				} else if (constituentMap.get(Type.ADVERBIAL).getClass() == IndexedConstituent.class) {
					IndexedConstituent c = (IndexedConstituent) (constituentMap.get(Type.ADVERBIAL));
					if (c.isPrepPhrase()) {
						if (c.getDependent() != null) {
							st = new SentenceTuple(constituentMap.get(Type.SUBJECT).rootString(),
									getVerbform(constituentMap.get(Type.VERB)), directObject, c.rootString(),
									c.getDependent().originalText());
						} else {
							st = new SentenceTuple(constituentMap.get(Type.SUBJECT).rootString(),
									getVerbform(constituentMap.get(Type.VERB)), directObject, c.rootString(), null);
						}
					} else {
						st = new SentenceTuple(constituentMap.get(Type.SUBJECT).rootString(),
								getVerbform(constituentMap.get(Type.VERB)), directObject, null, null);
					}
				} else {
					st = new SentenceTuple(constituentMap.get(Type.SUBJECT).rootString(),
							getVerbform(constituentMap.get(Type.VERB)), directObject,
							constituentMap.get(Type.ADVERBIAL).rootString(), null);
				}

				if (st != null)
					tuples.add(st);
			}
				break;
			case SVC: {
				String complement = null;

				if (constituentMap.containsKey(Type.COMPLEMENT)) {
					complement = constituentMap.get(Type.COMPLEMENT).rootString();
				} else if (constituentMap.containsKey(Type.ACOMP)) {
					complement = constituentMap.get(Type.ACOMP).rootString();
				} else if (constituentMap.containsKey(Type.CCOMP)) {
					complement = constituentMap.get(Type.CCOMP).rootString();
				} else if (constituentMap.containsKey(Type.XCOMP)) {
					complement = constituentMap.get(Type.XCOMP).rootString();
				}

				SentenceTuple st = new SentenceTuple(constituentMap.get(Type.SUBJECT).rootString(),
						getVerbform(constituentMap.get(Type.VERB)), null, null, complement);
				tuples.add(st);
			}
				break;
			case SVOC: {
				String complement = null;

				if (constituentMap.containsKey(Type.COMPLEMENT)) {
					complement = constituentMap.get(Type.COMPLEMENT).rootString();
				} else if (constituentMap.containsKey(Type.ACOMP)) {
					complement = constituentMap.get(Type.ACOMP).rootString();
				} else if (constituentMap.containsKey(Type.CCOMP)) {
					complement = constituentMap.get(Type.CCOMP).rootString();
				} else if (constituentMap.containsKey(Type.XCOMP)) {
					complement = constituentMap.get(Type.XCOMP).rootString();
				}

				String directObject = null;

				if (constituentMap.containsKey(Type.DOBJ)) {
					directObject = constituentMap.get(Type.DOBJ).rootString();
				}

				SentenceTuple st = new SentenceTuple(constituentMap.get(Type.SUBJECT).rootString(),
						getVerbform(constituentMap.get(Type.VERB)), directObject, null, complement);
				tuples.add(st);
			}
				break;
			case SVOO: {

				String directObject = null;

				if (constituentMap.containsKey(Type.DOBJ)) {
					directObject = constituentMap.get(Type.DOBJ).rootString();
				}

				String indirectObject = null;

				if (constituentMap.containsKey(Type.IOBJ)) {
					directObject = constituentMap.get(Type.IOBJ).rootString();
				}

				SentenceTuple st = new SentenceTuple(constituentMap.get(Type.SUBJECT).rootString(),
						getVerbform(constituentMap.get(Type.VERB)), directObject, null, indirectObject);
				tuples.add(st);
			}
				break;
			case UNKNOWN:
				break;
			default:
				break;

			}
		}

		if (retry) {
			// No clause detected, the reason could be parser's mistake
			// Or it could be that the description is fragmented
			if (clauses.size() == 0) {

				// Sometimes it's just that the sentence start with someone is not a trained much
				String modifiedSentence = sentence.replaceAll("someone", "he");

				if (!modifiedSentence.equals(sentence)) {
					List<SentenceTuple> retries = process(modifiedSentence, false, false);

					if (retries.size() > 0) {
						return retries;
					}
				}

				Tree t = clausIE.getDepTree().getChildrenAsList().get(0);

				// Verb phrase without subject
				if (t.label().toString().equals("S")) {
					modifiedSentence = "He " + sentence;

					List<SentenceTuple> retries = process(modifiedSentence, false, false);

					if (retries.size() > 0) {
						// Just remove the fake subject
						retries = retries.stream().map((SentenceTuple st) -> new SentenceTuple(null, st.subject,
								st.object, st.prep, st.prepDep)).collect(Collectors.toList());
						return retries;
					}
				}

				// Noun phrase
				// (ROOT
				// 		(NP
				// 			(NP (NN Dog)
				if (t.label().toString().equals("NP")) {
					List<Tree> grandchildren = t.getChildrenAsList();
					if (grandchildren.get(0).label().toString().equals("NP")) {
						modifiedSentence = StanfordUtils.getStringForm(grandchildren.get(0));

						modifiedSentence += " is ";

						modifiedSentence += StringUtils.join(
								grandchildren.subList(1, grandchildren.size()).stream()
										.map((Tree subtree) -> StanfordUtils.getStringForm(subtree)).collect(Collectors.toList()),
								" ");
						
						
						List<SentenceTuple> retries = process(modifiedSentence, false, false);

						if (retries.size() > 0) {
							// Just remove the fake subject
							retries = retries.stream().map((SentenceTuple st) -> new SentenceTuple(st.subject, null,
									st.object, st.prep, st.prepDep)).collect(Collectors.toList());
							return retries;
						}
					}
				}

				// Fragment phrase, could be just a prepositional phrase
				
				// (ROOT
				// 		(FRAG
				// 			(PP (IN Over)
				if (t.label().toString().equals("FRAG")) {
					List<Tree> grandchildren = t.getChildrenAsList();
					
					for (Tree grandchild : grandchildren) {
						if (grandchild.label().toString().equals("PP")) {
							modifiedSentence = "He is ";

							modifiedSentence += StanfordUtils.getStringForm(grandchild);
							
							List<SentenceTuple> retries = process(modifiedSentence, false, false);

							if (retries.size() > 0) {
								// Just remove the fake subject
								retries = retries.stream().map((SentenceTuple st) -> new SentenceTuple(null, null,
										st.object, st.prep, st.prepDep)).collect(Collectors.toList());
								return retries;
							}
						}
					}
				}
			}
		}

		return tuples;
	}

	public String getVerbform(Constituent verbConstituent) {
		if (verbConstituent.getClass() == IndexedConstituent.class) {
			IndexedConstituent c = (IndexedConstituent) verbConstituent;

			String s = c.rootString();
			for (IndexedWord iw : c.getSemanticGraph().getChildrenWithReln(c.getRoot(),
					EnglishGrammaticalRelations.PHRASAL_VERB_PARTICLE)) {
				s += " " + iw.originalText();
			}

			return s;
		}
		return verbConstituent.rootString();
	}

}
