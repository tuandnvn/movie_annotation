import java.io.IOException;

import de.mpii.clausie.ClausIE;
import de.mpii.clausie.Clause;
import de.mpii.clausie.Proposition;
import tuandn.ProcessSentence;
import tuandn.SentenceTuple;

public class Example {

	public static void main(String[] args) throws IOException {

		ClausIE clausIE = new ClausIE();
		clausIE.initParser();
		clausIE.getOptions().print(System.out, "# ");

		ProcessSentence ps = new ProcessSentence(clausIE);

		// input sentence
		// String sentence = "Bell, a telecommunication company, which is based in Los Angeles, makes and distributes
		// electronic, computer and building products.";

		// String[] sentences = new String[] { "Truffles picked during the spring are tasty.", "I start eating meat" ,
		// "Bill tried to shoot, demonstrating his incompetence",
		// "I want to eat then I want to go shopping", "He draws the gun and shoots", "We saw a man walking to a
		// house."};

		String[] sentences = new String[] { "SOMEONE walks and then stops.",
				"Altogether, his face dripping with sweat.",
				"Later, SOMEONE turns to two younger Marines hanging out by the car next to his.",
				"One raises his sports drink in a toast.",
				"SOMEONE takes a bag of ice from a cooler, raises it in reply, then holds the ice to his knee.",
				"A balding Marine, ices his neck.", "SOMEONE holds up a document.",
				"SOMEONE eyes SOMEONE's signed request.", "SOMEONE stares stoically.", "SOMEONE gazes at him.",
				"SOMEONE's eyes flicker to his friend's face.",
				"SOMEONE gives him a kind look, then adds his signature to the form.",
				"He closes it inside a service record folder and sets it in front of SOMEONE.",
				"There are three glasses of water on the TV set.", "there are people milling around" };
		// String sentence = "There is a ghost in the room";
		// sentence = "Bell sometimes makes products";
		// sentence = "By using its experise, Bell made great products in 1922 in Saarland.";
		// sentence = "Albert Einstein remained in Princeton.";
		// String sentence = "One raises his sports drink in a toast.";
		// String sentence = " Bell makes electronic, computer and building products.";

		for (String sentence : sentences) {
			System.out.println("Input sentence   : " + sentence);

			// parse tree
			System.out.print("Parse time       : ");
			long start = System.currentTimeMillis();
			clausIE.parse(sentence);
			long end = System.currentTimeMillis();
			System.out.println((end - start) / 1000. + "s");
			System.out.print("Dependency parse : ");
			System.out.println(clausIE.getDepTree().pennString().replaceAll("\n", "\n                   ").trim());
			System.out.print("Semantic graph   : ");
			System.out.println(
					clausIE.getSemanticGraph().toFormattedString().replaceAll("\n", "\n                   ").trim());

			// clause detection
			System.out.print("ClausIE time     : ");
			start = System.currentTimeMillis();
			clausIE.detectClauses();
			clausIE.generatePropositions();
			end = System.currentTimeMillis();
			System.out.println((end - start) / 1000. + "s");
			System.out.print("Clauses          : ");
			String sep = "";
			for (Clause clause : clausIE.getClauses()) {
				System.out.println(sep + clause.toString(clausIE.getOptions()));
				sep = "                   ";
			}

			// generate propositions
			System.out.print("Propositions     : ");
			sep = "";
			for (Proposition prop : clausIE.getPropositions()) {
				System.out.println(sep + prop.toString());
				sep = "                   ";
			}
			
			for (SentenceTuple st : ps.process(sentence)) {
				System.out.println(st);
			}
		}

	}
}
