
package jsat.text.tokenizer;

import java.io.Serializable;
import java.util.List;

/**
 * Interface for taking the text of a document and breaking it up into features. For example "This doc" might become "this" and "doc"
 * @author Edward Raff
 */
public interface Tokenizer extends Serializable
{
    public List<String> tokenize(String input);
}
