package tuandn;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import de.mpii.clausie.ClausIE;
import de.mpii.clausie.Clause;
import de.mpii.clausie.Constituent;
import de.mpii.clausie.IndexedConstituent;
import de.mpii.clausie.Constituent.Type;

public class ProcessSentence {
	ClausIE clausIE;

	public ProcessSentence(ClausIE clausIE) {
		this.clausIE = clausIE;
	}

	public List<SentenceTuple> process(String sentence) {
		List<SentenceTuple> tuples = new Vector<>();

		this.clausIE.parse(sentence);
		this.clausIE.detectClauses();
		this.clausIE.generatePropositions();

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
			case SV: {
				SentenceTuple st = new SentenceTuple(constituentMap.get(Type.SUBJECT).rootString(),
						constituentMap.get(Type.VERB).rootString(), null, null, null);
				// This still need to improved, for example:
				// The woman next to the table stands up
				// -> The womand stands up next to the table
				// We could use the prepositional phrase embedded in subject to replace the real prepositional phrase
				tuples.add(st);
			}
				break;
			case SVA: {
				if (constituentMap.get(Type.ADVERBIAL).getClass() == IndexedConstituent.class) {
					IndexedConstituent c = (IndexedConstituent) (constituentMap.get(Type.ADVERBIAL));
					if (c.isPrepPhrase()) {
						SentenceTuple st = new SentenceTuple(constituentMap.get(Type.SUBJECT).rootString(),
								constituentMap.get(Type.VERB).rootString(), null, c.rootString(),
								c.getDependent().originalText());
						tuples.add(st);
					}
				}

			}
				break;
			case SVC: {
				SentenceTuple st = new SentenceTuple(constituentMap.get(Type.SUBJECT).rootString(),
						constituentMap.get(Type.VERB).rootString(), null, null,
						constituentMap.get(Type.COMPLEMENT).rootString());
				tuples.add(st);
			}
				break;
			case SVO:
			case SVOA: {
				SentenceTuple st = null;

				if (!constituentMap.containsKey(Type.ADVERBIAL)) {
					st = new SentenceTuple(constituentMap.get(Type.SUBJECT).rootString(),
							constituentMap.get(Type.VERB).rootString(), constituentMap.get(Type.DOBJ).rootString(),
							null, null);
				} else if (constituentMap.get(Type.ADVERBIAL).getClass() == IndexedConstituent.class) {
					IndexedConstituent c = (IndexedConstituent) (constituentMap.get(Type.ADVERBIAL));
					if (c.isPrepPhrase()) {
						st = new SentenceTuple(constituentMap.get(Type.SUBJECT).rootString(),
								constituentMap.get(Type.VERB).rootString(), constituentMap.get(Type.DOBJ).rootString(),
								c.rootString(), c.getDependent().originalText());
					} else {
						st = new SentenceTuple(constituentMap.get(Type.SUBJECT).rootString(),
								constituentMap.get(Type.VERB).rootString(), constituentMap.get(Type.DOBJ).rootString(),
								c.rootString(), null);
					}
				} else {
					st = new SentenceTuple(constituentMap.get(Type.SUBJECT).rootString(),
							constituentMap.get(Type.VERB).rootString(), constituentMap.get(Type.DOBJ).rootString(),
							constituentMap.get(Type.ADVERBIAL).rootString(), null);
				}

				tuples.add(st);
			}
				break;
			case SVOC: {
				SentenceTuple st = new SentenceTuple(constituentMap.get(Type.SUBJECT).rootString(),
						constituentMap.get(Type.VERB).rootString(), constituentMap.get(Type.DOBJ).rootString(), null,
						constituentMap.get(Type.COMPLEMENT).rootString());
				tuples.add(st);
			}
				break;
			case SVOO: {
				SentenceTuple st = new SentenceTuple(constituentMap.get(Type.SUBJECT).rootString(),
						constituentMap.get(Type.VERB).rootString(), constituentMap.get(Type.DOBJ).rootString(), null,
						constituentMap.get(Type.IOBJ).rootString());
				tuples.add(st);
			}
				break;
			case UNKNOWN:
				break;
			default:
				break;

			}
		}

		return tuples;
	}
}
