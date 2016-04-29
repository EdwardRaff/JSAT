package jsat.text;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
import jsat.text.tokenizer.Tokenizer;
import jsat.text.wordweighting.WordWeighting;

/**
 * Creates new text vectors from a dictionary of known tokens and a word 
 * weighting scheme. <br>
 * <br>
 * This object is generally intended to be constructed by a
 * {@link TextDataLoader}, though can be used if you know all the words you will
 * need (and can initialize the {@link WordWeighting}) before creating this
 * object.
  
 * 
 * @author Edward Raff
 */
public class BasicTextVectorCreator implements TextVectorCreator
{

    private static final long serialVersionUID = -8620485679300539556L;
    private final Tokenizer tokenizer;
    private final Map<String, Integer> wordIndex;
    private final WordWeighting weighting;

    /**
     * Creates a new basic text vector creator
     * @param tokenizer the tokenizer to apply to incoming strings
     * @param wordIndex the map of each known word to its index, the size of the
     * map indicating the maximum (exclusive) index
     * @param weighting the weighting process to apply to each loaded document.
     * This should have already been initialized, or be stateless.
     */
    public BasicTextVectorCreator(Tokenizer tokenizer, Map<String, Integer> wordIndex, WordWeighting weighting)
    {
        this.tokenizer = tokenizer;
        this.wordIndex = wordIndex;
        this.weighting = weighting;
    }

    @Override
    public Vec newText(String text)
    {
        return newText(text, new StringBuilder(), new ArrayList<String>());
    }

    @Override
    public Vec newText(String input, StringBuilder workSpace, List<String> storageSpace)
    {
        tokenizer.tokenize(input, workSpace, storageSpace);
        SparseVector vec = new SparseVector(wordIndex.size());
        for( String word : storageSpace)
        {
            if(wordIndex.containsKey(word))//Could also call retainAll on words before looping. Worth while to investigate 
            {
                int index = wordIndex.get(word);
                vec.increment(index, 1.0);
            }
        }
        
        weighting.applyTo(vec);
        return vec;
    }
}
