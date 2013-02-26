package jsat.text;

import java.io.Serializable;
import java.util.List;
import jsat.linear.Vec;

/**
 * A Text Vector Creator is an object that can convert a text string into a 
 * {@link Vec}
 * 
 * @author Edward Raff
 */
public interface TextVectorCreator extends Serializable
{
    /**
     * Converts the given input text into a vector representation. 
     * @param input the input string
     * @return a vector representation
     */
    public Vec newText(String input);
    
    /**
     * Converts the given input text into a vector representation
     * @param input the input string
     * @param workSpace an already allocated (but empty) string builder than can
     * be used as a temporary work space. 
     * @param storageSpace an already allocated (but empty) list to place the 
     * tokens into
     * @return a vector representation
     */
    public Vec newText(String input, StringBuilder workSpace, List<String> storageSpace);
}
