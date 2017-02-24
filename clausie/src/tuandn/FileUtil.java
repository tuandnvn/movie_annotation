package tuandn;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

public class FileUtil {

	public static String readFile(String fileName) throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(fileName)), "UTF-8"));
		try {
			StringBuilder sb = new StringBuilder();
			String line = br.readLine();

			while (line != null) {
				sb.append(line);
				sb.append("\n");
				line = br.readLine();
			}
			return sb.toString();
		} finally {
			br.close();
		}
	}

	public static byte[] readBytes(String fileName) throws IOException {
		Path path = Paths.get(fileName);
		return Files.readAllBytes(path);
	}

	public static void listFilePaths(String directoryName, ArrayList<StringBuilder> filePaths) {
		File directory = new File(directoryName);

		// get all the files from a directory
		File[] fList = directory.listFiles();
		for (File file : fList) {
			if (file.isFile()) {
				filePaths.add(new StringBuilder().append(file.getAbsolutePath()));
			} else if (file.isDirectory()) {
				listFilePaths(file.getAbsolutePath(), filePaths);
			}
		}
	}

	public static void writeToFile(InputStream inputStream, String fileName, boolean append) throws IOException {
		OutputStream outputStream = null;
		// Files.createDirectories(Paths.get(fileName).getParent());

		outputStream = new FileOutputStream(new File(fileName), append);

		int read = 0;
		byte[] buffer = new byte[1024];
		while ((read = inputStream.read(buffer)) != -1) {
			outputStream.write(buffer, 0, read);
		}
		inputStream.close();
		outputStream.close();
	}

	public static void writeObject(FileOutputStream fout, Serializable object) throws IOException {
		ObjectOutputStream oos = null;
		oos = new ObjectOutputStream(fout);
		oos.writeObject(object);
		oos.flush();
		oos.close();
	}

	public static void writeString(String s, String fileName) throws IOException {
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File(fileName)));
		bw.write(s);
		bw.close();
	}

	public static Object readObject(InputStream streamIn) throws IOException, ClassNotFoundException {
		ObjectInputStream objectinputstream = null;
		objectinputstream = new ObjectInputStream(streamIn);
		Object result = objectinputstream.readObject();
		objectinputstream.close();
		return result;
	}

	public static String readString(InputStream streamIn) throws IOException {
		BufferedReader reader = new BufferedReader(new InputStreamReader(streamIn));
		String line = null;
		StringBuilder stringBuilder = new StringBuilder();
		while ((line = reader.readLine()) != null) {
			stringBuilder.append(line);
			stringBuilder.append("\n");
		}
		return stringBuilder.toString();
	}
}