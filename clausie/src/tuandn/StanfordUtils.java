package tuandn;

import edu.stanford.nlp.trees.Tree;

public class StanfordUtils {
	public static String getStringForm(Tree tree) {
		if (tree.isLeaf()) {
			return tree.nodeString();
		}

		StringBuilder sb = new StringBuilder();

		for (Tree t : tree.getChildrenAsList()) {
			sb.append(getStringForm(t));
			sb.append(" ");
		}

		return sb.toString().trim();
	}
}
