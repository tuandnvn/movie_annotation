package tuandn;

/**
 * This is a simple representation of a clause
 * For example, clause: 
 * SOMEONE takes a bag of ice from a cooler
 * @author Tuan
 *
 */
public class SentenceTuple {
	public String subject;
	public String verb;
	public String object;
	public String prep;
	public String prepDep;

	public SentenceTuple(String subject, String verb, String object, String prep, String prepDep) {
		super();
		this.subject = subject;
		this.verb = verb;
		this.object = object;
		this.prep = prep;
		this.prepDep = prepDep;
	}

	@Override
	public String toString() {
		return "SentenceTuple [subject=" + subject + ", verb=" + verb + ", object=" + object + ", prep=" + prep
				+ ", prepDep=" + prepDep + "]";
	}

	/**
	 * I will have to produce a sentence form string from this representation
	 * @return
	 */
	public String toSentenceString() {
		return "";
	}
}
