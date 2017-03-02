package tuandn;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import java.util.stream.Collectors;

import org.apache.commons.lang3.StringUtils;

import de.mpii.clausie.ClausIE;
import de.mpii.clausie.Clause;
import de.mpii.clausie.Constituent;
import de.mpii.clausie.Constituent.Type;
import de.mpii.clausie.IndexedConstituent;
import de.mpii.clausie.TextConstituent;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.EnglishGrammaticalRelations;
import edu.stanford.nlp.trees.Tree;

public class ProcessSentence {
	ClausIE clausIE;

	public ProcessSentence(ClausIE clausIE) {
		this.clausIE = clausIE;
	}

	// Make some small change to the original sentence
	// Adding a subject "he" after "Then"
	// Change all SOMEONE to HE
	public String preprocess(String sentence) {
		sentence = sentence.replaceAll("SOMEONE", "HE");
		sentence = sentence.replaceAll(" then ", " then he ");
		sentence = sentence.replaceAll("Then ", "Then he ");
		sentence = sentence.replaceAll(",then ", ",then he ");
		return sentence;
	}

	public static String replaceHEtosomeone(String s) {
		if (s == null)
			return null;
		if (s.equals("HE"))
			return "someone";
		return s;
	}

	public static SentenceTuple replaceHEtosomeone(SentenceTuple st) {
		String subject = replaceHEtosomeone(st.subject);
		String verb = replaceHEtosomeone(st.verb);
		String object = replaceHEtosomeone(st.object);
		String prep = replaceHEtosomeone(st.prep);
		String prepDep = replaceHEtosomeone(st.prepDep);
		return new SentenceTuple(subject, verb, object, prep, prepDep);
	}

	public List<SentenceTuple> postprocess(List<SentenceTuple> sentenceTuples) {
		List<SentenceTuple> results = sentenceTuples.stream().filter((SentenceTuple st) -> !st.equalNone())
				.map((SentenceTuple st) -> replaceHEtosomeone(st)).collect(Collectors.toList());

		return results;
	}

	public List<SentenceTuple> process(String sentence, boolean verbose, boolean retry) {
		sentence = preprocess(sentence);
		List<SentenceTuple> tuples = normalProcess(sentence, verbose, retry);
		return postprocess(tuples);
	}

	public List<SentenceTuple> normalProcess(String sentence, boolean verbose, boolean retry) {
		String[] replaces = new String[] { "Later", "Now", "Back" };
		// Remove all temporal adverb
		for (String replace : replaces) {
			sentence = sentence.replaceAll(replace + ",", "");
			sentence = sentence.replaceAll(replace, "");
		}
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
			case EXISTENTIAL: {
				SentenceTuple st = null;

				String subject = null;
				if (constituentMap.containsKey(Type.SUBJECT)) {
					subject = getNounForm(constituentMap.get(Type.SUBJECT));
				}

				// There is still chance that directObject is not null
				String directObject = null;

				if (constituentMap.containsKey(Type.DOBJ)) {
					directObject = getNounForm(constituentMap.get(Type.DOBJ));
				}

				if (!constituentMap.containsKey(Type.ADVERBIAL)) {
					st = new SentenceTuple(subject, null, directObject, null, null);
				} else if (constituentMap.get(Type.ADVERBIAL).getClass() == IndexedConstituent.class) {
					IndexedConstituent c = (IndexedConstituent) (constituentMap.get(Type.ADVERBIAL));

					if (c.isPrepPhrase() && c.getDependent() != null) {
						st = new SentenceTuple(subject, null, directObject, getPrepositionalForm(c),
								getDependentForm(c));
					} else {
						st = new SentenceTuple(subject, null, directObject, c.rootString(), null);
					}
				} else if (constituentMap.get(Type.ADVERBIAL).getClass() == TextConstituent.class) {
					st = new SentenceTuple(subject, null, directObject, constituentMap.get(Type.ADVERBIAL).rootString(),
							null);
				}

				if (st != null)
					tuples.add(st);
			}
				break;
			case SV:
			case SVA: {
				SentenceTuple st = null;

				String subject = null;
				if (constituentMap.containsKey(Type.SUBJECT)) {
					subject = getNounForm(constituentMap.get(Type.SUBJECT));
				}

				// There is still chance that directObject is not null
				String directObject = null;

				if (constituentMap.containsKey(Type.DOBJ)) {
					directObject = getNounForm(constituentMap.get(Type.DOBJ));
				}

				String verb = getVerbform(constituentMap.get(Type.VERB));

				// This still need to improved, for example:
				// The woman next to the table stands up
				// -> The womand stands up next to the table
				// We could use the prepositional phrase embedded in subject to replace the real prepositional phrase
				if (!constituentMap.containsKey(Type.ADVERBIAL)) {
					st = new SentenceTuple(subject, verb, directObject, null, null);
				} else if (constituentMap.get(Type.ADVERBIAL).getClass() == IndexedConstituent.class) {
					IndexedConstituent c = (IndexedConstituent) (constituentMap.get(Type.ADVERBIAL));

					if (c.isPrepPhrase() && c.getDependent() != null) {
						st = new SentenceTuple(subject, verb, directObject, getPrepositionalForm(c),
								getDependentForm(c));
					} else {
						st = new SentenceTuple(subject, verb, directObject, c.rootString(), null);
					}
				} else if (constituentMap.get(Type.ADVERBIAL).getClass() == TextConstituent.class) {
					st = new SentenceTuple(subject, verb, directObject, constituentMap.get(Type.ADVERBIAL).rootString(),
							null);
				}

				if (st != null)
					tuples.add(st);
			}
				break;

			case SVO:
			case SVOA: {
				SentenceTuple st = null;

				String subject = null;
				if (constituentMap.containsKey(Type.SUBJECT)) {
					subject = getNounForm(constituentMap.get(Type.SUBJECT));
				}

				String directObject = null;

				if (constituentMap.containsKey(Type.DOBJ)) {
					directObject = getNounForm(constituentMap.get(Type.DOBJ));
				}

				String verb = getVerbform(constituentMap.get(Type.VERB));

				if (!constituentMap.containsKey(Type.ADVERBIAL)) {
					st = new SentenceTuple(subject, verb, directObject, null, null);
				} else if (constituentMap.get(Type.ADVERBIAL).getClass() == IndexedConstituent.class) {
					IndexedConstituent c = (IndexedConstituent) (constituentMap.get(Type.ADVERBIAL));
					if (c.isPrepPhrase() && c.getDependent() != null) {
						st = new SentenceTuple(subject, verb, directObject, getPrepositionalForm(c),
								getDependentForm(c));
					} else {
						st = new SentenceTuple(subject, verb, directObject, c.rootString(), null);
					}
				} else {
					st = new SentenceTuple(subject, verb, directObject, constituentMap.get(Type.ADVERBIAL).rootString(),
							null);
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

				SentenceTuple st = new SentenceTuple(getNounForm(constituentMap.get(Type.SUBJECT)),
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
					directObject = getNounForm(constituentMap.get(Type.DOBJ));
				}

				SentenceTuple st = new SentenceTuple(getNounForm(constituentMap.get(Type.SUBJECT)),
						getVerbform(constituentMap.get(Type.VERB)), directObject, null, complement);
				tuples.add(st);
			}
				break;
			case SVOO: {

				String directObject = null;

				if (constituentMap.containsKey(Type.DOBJ)) {
					directObject = getNounForm(constituentMap.get(Type.DOBJ));
				}

				String indirectObject = null;

				if (constituentMap.containsKey(Type.IOBJ)) {
					directObject = constituentMap.get(Type.IOBJ).rootString();
				}

				SentenceTuple st = new SentenceTuple(getNounForm(constituentMap.get(Type.SUBJECT)),
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
					tuples = process(modifiedSentence, false, false);
				}

				if (tuples.size() == 0) {
					Tree t = clausIE.getDepTree().getChildrenAsList().get(0);

					// Verb phrase without subject
					if (t.label().toString().equals("S")) {
						modifiedSentence = "He " + sentence;

						tuples = process(modifiedSentence, false, false);

						if (tuples.size() > 0) {
							// Just remove the fake subject
							tuples = tuples.stream().map((SentenceTuple st) -> new SentenceTuple(null, st.verb,
									st.object, st.prep, st.prepDep)).collect(Collectors.toList());
						}
					}

					// Noun phrase
					// (ROOT
					// (NP
					// (NP (NN Dog)
					else if (t.label().toString().equals("NP")) {
						List<Tree> grandchildren = t.getChildrenAsList();
						if (grandchildren.get(0).label().toString().equals("NP")) {
							modifiedSentence = StanfordUtils.getStringForm(grandchildren.get(0));

							modifiedSentence += " is ";

							modifiedSentence += StringUtils.join(grandchildren.subList(1, grandchildren.size()).stream()
									.map((Tree subtree) -> StanfordUtils.getStringForm(subtree))
									.collect(Collectors.toList()), " ");

							tuples = process(modifiedSentence, false, false);

							if (tuples.size() > 0) {
								// Just remove the fake subject
								tuples = tuples.stream().map((SentenceTuple st) -> new SentenceTuple(st.subject, null,
										st.object, st.prep, st.prepDep)).collect(Collectors.toList());
							}
						}
					}

					// Fragment phrase, could be just a prepositional phrase

					// (ROOT
					// (FRAG
					// (PP (IN Over)
					else if (t.label().toString().equals("FRAG")) {
						List<Tree> grandchildren = t.getChildrenAsList();

						// (FRAG
						// (PP (IN Over)

						// or
						// (FRAG
						// (ADVP (RB Back)
						// (PP (IN in)
						// (NP (DT the) (NN kitchen))))
						// (. .)))
						for (Tree grandchild : grandchildren) {
							if (grandchild.label().toString().equals("PP")
									|| grandchild.label().toString().equals("ADVP")) {
								modifiedSentence = "He eat ";

								modifiedSentence += StanfordUtils.getStringForm(grandchild);

								tuples = process(modifiedSentence, false, false);

								if (tuples.size() > 0) {
									// Just remove the fake subject
									tuples = tuples.stream().map((SentenceTuple st) -> new SentenceTuple(null, null,
											st.object, st.prep, st.prepDep)).collect(Collectors.toList());
								}
							}

						}
					}
				}

			}
		}

		if (tuples.size() == 1) {
			adjustPrep(tuples);
		}

		return tuples;
	}

	public String getNounForm(Constituent constituent) {
		if (constituent.getClass() == TextConstituent.class)
			return constituent.rootString();

		if (constituent.getClass() == IndexedConstituent.class) {
			IndexedConstituent c = (IndexedConstituent) constituent;

			String[] headwords = new String[] { "some", "each", "most", "one", "two", "three", "four", "five", "six",
					"seven", "eight", "nine", "ten", "dozens", "dozen", "group", "mass" };
			for (String headword : headwords) {
				if (c.rootString().toLowerCase().equals(headword)) {
					if (c.getSemanticGraph().getChildList(c.getRoot()).size() != 0) {
						for (IndexedWord PPHead : c.getSemanticGraph().getChildList(c.getRoot())) {
							if (PPHead.originalText().equals("of")) {
								IndexedWord NPHead = c.getSemanticGraph().getChildList(PPHead).get(0);
								return getTextForm(NPHead);
							}
						}
					}
				}
			}

			return c.rootString();
		}
		return null;
	}

	public void adjustPrep(List<SentenceTuple> tuples) {
		// No preposition
		// Try recover from other place
		if (tuples.get(0).prep == null && tuples.get(0).prepDep == null) {
			SemanticGraph sg = clausIE.getSemanticGraph();

			// Search for PP
			List<IndexedWord> ts = sg.getChildList(sg.getFirstRoot());
			while (!ts.isEmpty()) {
				IndexedWord iw = ts.remove(0);

				for (IndexedWord child : sg.getChildList(iw)) {
					if (sg.getEdge(iw, child).getRelation() == EnglishGrammaticalRelations.PREPOSITIONAL_MODIFIER) {
						tuples.get(0).prep = getTextForm(child);

						for (IndexedWord grandchild : sg.getChildList(child)) {
							if (sg.getEdge(child, grandchild)
									.getRelation() == EnglishGrammaticalRelations.PREPOSITIONAL_OBJECT) {
								tuples.get(0).prepDep = getTextForm(grandchild);

								return;
							}
						}
						return;
					}
				}

				ts.addAll(sg.getChildList(iw));
			}
		}
	}

	public String getPrepositionalForm(IndexedConstituent c) {
		if (c.rootString().equals("in") && c.getDependent() != null && c.getDependent().originalText().equals("front"))
			return "in front of";

		if (c.rootString().equals("at") && c.getDependent() != null && c.getDependent().originalText().equals("rear"))
			return "at the rear of";

		if (c.rootString().equals("on") && c.getDependent() != null && c.getDependent().originalText().equals("side"))
			return "on the side of";

		return c.rootString();
	}

	public String getDependentForm(IndexedConstituent c) {
		if (c.rootString().equals("in") && c.getDependent().originalText().equals("front")) {
			if (c.getSemanticGraph().getChildList(c.getDependent()).size() != 0) {
				for (IndexedWord PPHead : c.getSemanticGraph().getChildList(c.getDependent())) {
					if (PPHead.originalText().equals("of")) {
						IndexedWord NPHead = c.getSemanticGraph().getChildList(PPHead).get(0);
						return getTextForm(NPHead);
					}
				}
			}
		}

		if (c.rootString().equals("at") && c.getDependent().originalText().equals("rear")) {
			if (c.getSemanticGraph().getChildList(c.getDependent()).size() != 0) {
				for (IndexedWord PPHead : c.getSemanticGraph().getChildList(c.getDependent())) {
					if (PPHead.originalText().equals("of")) {
						IndexedWord NPHead = c.getSemanticGraph().getChildList(PPHead).get(0);
						return getTextForm(NPHead);
					}
				}
			}
		}

		if (c.rootString().equals("on") && c.getDependent().originalText().equals("side")) {
			if (c.getSemanticGraph().getChildList(c.getDependent()).size() != 0) {
				for (IndexedWord PPHead : c.getSemanticGraph().getChildList(c.getDependent())) {
					if (PPHead.originalText().equals("of")) {
						IndexedWord NPHead = c.getSemanticGraph().getChildList(PPHead).get(0);
						return getTextForm(NPHead);
					} else {
						return "";
					}
				}
			}
		}

		if (c.getDependent().originalText().equals("back")) {
			if (c.getSemanticGraph().getChildList(c.getDependent()).size() != 0) {
				for (IndexedWord PPHead : c.getSemanticGraph().getChildList(c.getDependent())) {
					if (PPHead.originalText().equals("of")) {
						IndexedWord NPHead = c.getSemanticGraph().getChildList(PPHead).get(0);
						return getTextForm(NPHead);
					} else {
						return "";
					}
				}
			}
		}

		return c.getDependent().originalText();

	}

	public String getTextForm(IndexedWord word) {
		return word.originalText();
	}

	public String getVerbform(Constituent verbConstituent) {
		if (verbConstituent.getClass() == IndexedConstituent.class) {
			IndexedConstituent c = (IndexedConstituent) verbConstituent;

			String s = c.rootString();
			for (IndexedWord iw : c.getSemanticGraph().getChildrenWithReln(c.getRoot(),
					EnglishGrammaticalRelations.PHRASAL_VERB_PARTICLE)) {
				s += " " + getTextForm(iw);
			}

			return s;
		}
		return verbConstituent.rootString();
	}

}
