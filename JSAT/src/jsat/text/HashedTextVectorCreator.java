package jsat.text;

import java.util.ArrayList;
import java.util.List;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
import jsat.text.tokenizer.Tokenizer;
import jsat.text.wordweighting.WordWeighting;

/**
 * Text Vector creator to that uses hashed features. 
 * 
 * @author Edward Raff
 */
public class HashedTextVectorCreator implements TextVectorCreator
{
    private int dimensionSize;
    private Tokenizer tokenizer;
    private WordWeighting weighting;

    /**
     * Creates a new text vector creator that works with hash-trick features
     * @param dimensionSize the dimension size of the feature space
     * @param tokenizer the tokenizer to apply to incoming strings
     * @param weighting the weighting process to apply to each loaded document. 
     */
    public HashedTextVectorCreator(int dimensionSize, Tokenizer tokenizer, WordWeighting weighting)
    {
        if(dimensionSize <= 1)
            throw new ArithmeticException("Vector dimension must be a positive value");
        this.dimensionSize = dimensionSize;
        this.tokenizer = tokenizer;
        this.weighting = weighting;
    }

    @Override
    public Vec newText(String input)
    {
        return newText(input, new StringBuilder(), new ArrayList<String>());
    }

    @Override
    public Vec newText(String input, StringBuilder workSpace, List<String> storageSpace)
    {
        tokenizer.tokenize(input, workSpace, storageSpace);
        SparseVector vec = new SparseVector(dimensionSize);
        for(String word : storageSpace)
            vec.increment(Math.abs(word.hashCode())%dimensionSize, 1.0);
        weighting.applyTo(vec);
        return vec;
    }
}
