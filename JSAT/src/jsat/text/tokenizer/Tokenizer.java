
package jsat.text.tokenizer;

import java.io.Serializable;
import java.util.List;

/**
 * Interface for taking the text of a document and breaking it up into features.
 * For example "This doc" might become "this" and "doc"
 * 
 * @author Edward Raff
 */
public interface Tokenizer extends Serializable
{
    /**
     * Breaks the input string into a series of tokens that may be used as 
     * features for a classifier. The returned tokens must be either new string
     * objects or interned strings. If a token is returned that is backed by 
     * the original document, memory may get leaked by processes consuming the
     * token. <br>
     * This method should be thread safe
     * 
     * @param input the string to tokenize
     * @return an already allocated list to place the tokens into
     */
    public List<String> tokenize(String input);
    
    /**
     * Breaks the input string into a series of tokens that may be used as 
     * features for a classifier. The returned tokens must be either new string
     * objects or interned strings. If a token is returned that is backed by 
     * the original document, memory may get leaked by processes consuming the
     * token. <br>
     * This method should be thread safe
     * 
     * @param input the string to tokenize
     * @param workSpace an already allocated (but empty) string builder than can
     * be used as a temporary work space. 
     * @param storageSpace an already allocated (but empty) list to place the 
     * tokens into
     */
    public void tokenize(String input, StringBuilder workSpace, List<String> storageSpace);
}
