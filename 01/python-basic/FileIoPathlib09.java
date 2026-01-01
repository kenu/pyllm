import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class FileIoPathlib09 {
    public static void main(String[] args) throws IOException {
        Path base = Paths.get("").toAbsolutePath();
        Path p = base.resolve("sample.txt");

        Files.writeString(p, "line1\nline2\n", StandardCharsets.UTF_8);

        String text = Files.readString(p, StandardCharsets.UTF_8);
        System.out.println(text.trim());

        List<String> lines = Files.readAllLines(p, StandardCharsets.UTF_8);
        System.out.println("lines: " + lines);

        Files.deleteIfExists(p);
    }
}
