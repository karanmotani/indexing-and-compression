import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Comparator;
import java.util.Date;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.TreeMap;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;


public class IndexCompression
{
	static PrintWriter out;
	static Stemmer stemmer = new Stemmer();
	static int DocumentLength[];
	static int maxTermFreq[];
	static int maxStemFreq[];
	static double uncompressedIndexTime = 0.0;
	static long compressedIndexTime1=0;
	static long compressedIndexTime2=0;
	static StanfordCoreNLP pipeline1;
	

	
//	------------------ Stop Words Extraction Function ------------------
	
	public static ArrayList<String> stopWordsExtraction(File stopWordFile) throws IOException
	{

		String line = "";
		ArrayList<String> stopWords = new ArrayList<String>();		
		
		FileReader fileReader = new FileReader(stopWordFile);
		//@SuppressWarnings("resource")
		BufferedReader br = new BufferedReader(fileReader);
		while ((line = br.readLine()) != null) 
			stopWords.add(line.trim());
		
		br.close();
		return stopWords;
		
	}
	

//	------------------ Term & Document Frequency Node ------------------
	
	public static class TermDocFreqNode {
		int tF;
		int dF;
		TreeMap<Integer,ArrayList<Integer>> postingList = new TreeMap<Integer, ArrayList<Integer>>();
		//TreeMap<Integer, Integer> postingFiles = new TreeMap<Integer, Integer>();
	}

	
//	------------------ Compressed Term & Document Frequency ------------------
	
	public static class TermDocFreqCompressedIndex 
	{
		static String frontCodeNode;
		byte tF[];
		byte dF[];
		int ptr;
		ArrayList<PostingListNode> postingList = new ArrayList<PostingListNode>();
	}
	
	
//	------------------ Positing List ------------------
	
	public static class PostingListNode
	{
		byte[] docID;
		byte[] tF;
		byte[] maxTF1;
		byte[] docLen;
	}
	
	
//	------------------ Calculating Gamma Code Function ------------------
	
	public static String calcGamma (Integer docID) 
	{
		int i;
		String gammaCode, unary = "";
		String bin = Integer.toBinaryString(docID);
		bin = bin.substring(1);
		
		for (i = 0; i < bin.length(); i++) 
			unary = unary + "1";
		
		unary = unary + "0";
		gammaCode = unary + bin;
		return gammaCode;
	}

//	------------------ Calculating Delta Code Function ------------------

	public static byte[] calcDelta (Integer tF) 
	{
		String delta, gamma;
		String bin = Integer.toBinaryString(tF);
		BitSet bitset = new BitSet();
		gamma = calcGamma(bin.length());
		bin = bin.substring(1);
		delta = gamma + bin;
		bitset = BitSet.valueOf(new long[] { Long.parseLong(delta, 2) });
		return bitset.toByteArray();
	}
	
	
//	------------------ Extracting Calculated Gamma Code Function ------------------
	
	public static byte[] calcGammaExtract (Integer docID)
	{
		BitSet bitset = new BitSet();
		bitset = BitSet.valueOf(new long[] { Long.parseLong(calcGamma(docID), 2) });
		return bitset.toByteArray();
	}
	
	
//	------------------ Calculate Compressed File using Delta Code Function ------------------
	
	public static ArrayList<PostingListNode> calcCompressedUsingDeltaCode (TreeMap<Integer, ArrayList<Integer>> postingList) 
	{

		ArrayList<PostingListNode> compList = new ArrayList<>();
		Iterator<Map.Entry<Integer, ArrayList<Integer>>> iterator = postingList.entrySet().iterator();
		Map.Entry<Integer, ArrayList<Integer>> entry = iterator.next();
		Integer entryDocIDOne = entry.getKey();
		Integer tF = (entry.getValue()).get(0);

		PostingListNode postingListNode = new PostingListNode();
		postingListNode.docID = calcDelta(entryDocIDOne);
		postingListNode.tF = calcDelta(tF);
		postingListNode.maxTF1 = calcDelta(maxTermFreq[entryDocIDOne.intValue()]);
		postingListNode.docLen = calcDelta(DocumentLength[entryDocIDOne.intValue()]);
		compList.add(postingListNode);

		while(iterator.hasNext()){
			entry = iterator.next();
			Integer entryDocID = entry.getKey();
			tF = (entry.getValue()).get(0);

			postingListNode = new PostingListNode();
			postingListNode.docID = calcDelta(entryDocID - entryDocIDOne);
			postingListNode.tF = calcDelta(tF);
			postingListNode.maxTF1 = calcDelta(maxTermFreq[entryDocID.intValue()]);
			postingListNode.docLen = calcDelta(DocumentLength[entryDocID.intValue()]);

			compList.add(postingListNode);
			entryDocIDOne = entryDocID;
		}
		return compList;
	
	}
	
	
//	------------------ Calculate Compressed File using Gamma Code Function ------------------
	
	public static ArrayList<PostingListNode> calcCompressedUsingGammaCode (TreeMap<Integer, ArrayList<Integer>> postingList) 
	{

		Iterator<Map.Entry<Integer, ArrayList<Integer>>> iterator = postingList.entrySet().iterator();
		Map.Entry<Integer, ArrayList<Integer>> entry = iterator.next();
		ArrayList<PostingListNode> compList = new ArrayList<>();
		Integer entryDocIDOne = entry.getKey();
		int tF = (entry.getValue()).get(0);

		PostingListNode postingListNode = new PostingListNode();
		postingListNode.tF = calcGammaExtract(tF);
		postingListNode.maxTF1 = calcGammaExtract(maxTermFreq[entryDocIDOne.intValue()]);
		postingListNode.docID = calcGammaExtract(entryDocIDOne);
		postingListNode.docLen = calcGammaExtract(DocumentLength[entryDocIDOne.intValue()]);
		compList.add(postingListNode);

		while(iterator.hasNext())
		{
			entry = iterator.next();
			Integer entryDocID = entry.getKey();
			tF = (entry.getValue()).get(0);

			postingListNode = new PostingListNode();
			postingListNode.docID = calcGammaExtract(entryDocID - entryDocIDOne);
			postingListNode.tF = calcGammaExtract(tF);
			postingListNode.maxTF1 = calcGammaExtract(maxTermFreq[entryDocID.intValue()]);
			postingListNode.docLen = calcGammaExtract(DocumentLength[entryDocID.intValue()]);

			compList.add(postingListNode);
			entryDocIDOne = entryDocID;
		}
		return compList;
		
	}
	
	
//	------------------ Calculating Compressed Posting List 1 Function ------------------
	
	public static TermDocFreqCompressedIndex postingListOneCompression(TreeMap<String, TermDocFreqNode> treeMap, String s) 
	{
		TermDocFreqNode termDocFreqNode = treeMap.get(s);
		TermDocFreqCompressedIndex termDocFreqCompressedIndex = new TermDocFreqCompressedIndex();
		termDocFreqCompressedIndex.postingList  = new ArrayList<>();
		
		termDocFreqCompressedIndex.tF = calcGammaExtract(termDocFreqNode.tF);
		termDocFreqCompressedIndex.dF = calcGammaExtract(termDocFreqNode.dF);
		termDocFreqCompressedIndex.postingList.addAll(calcCompressedUsingGammaCode(termDocFreqNode.postingList));
		
		return termDocFreqCompressedIndex;
	}
	
	
//	------------------ Calculating Compressed Posting List 2 Function ------------------
	
	public static TermDocFreqCompressedIndex postingListTwoCompression(TreeMap<String, TermDocFreqNode> treeMap, String s) 
	{
		TermDocFreqNode termDocFreqNode = treeMap.get(s);
		TermDocFreqCompressedIndex termDocFreqCompressedIndex = new TermDocFreqCompressedIndex();
		termDocFreqCompressedIndex.postingList  = new ArrayList<>();
		
		termDocFreqCompressedIndex.tF = calcDelta(termDocFreqNode.tF);
		termDocFreqCompressedIndex.dF = calcDelta(termDocFreqNode.dF);
		termDocFreqCompressedIndex.postingList.addAll(calcCompressedUsingDeltaCode(termDocFreqNode.postingList));
		
		return termDocFreqCompressedIndex;
	}
	
	
//	------------------ Calculating Block Coding Function ------------------
	
	public static String calcBlockCode(String s[]) 
	{
		int i;
		String temp1;
		String blockCode = "";
		for(i=0; i<s.length; i++)
		{
			temp1 = s[i];
			blockCode = blockCode + temp1.length() + temp1 ;
		}
		return blockCode;
	}
	
	
//	------------------ Calculating Front Coding Function ------------------
	
	public static String calcFrontCode(String s[]) 
	{
		int i;
		String temp1, temp2 = "", frontCode = "";
		temp1 = s[0];
		frontCode = frontCode + temp1.length() + temp1 + "*";
		
		for (i = 0; i < s.length; i++) 
		{
			temp2 = s[i];
			temp2 = temp2.replace(temp1, "");
			frontCode = frontCode + temp2.length() + temp2 + "*";
		}
		return frontCode;
	}
	
	
//	------------------ Sorting according to Frequency ------------------
	
	public static TreeMap<String, Integer> freqSort(final TreeMap<String, Integer> token)
	{
			Comparator<String> comparator = new Comparator<String>() 
			{
					public int compare(String string1, String string2) 
					{
							if (token.get(string2).compareTo(token.get(string1)) == 0)
									return 1;
							else
									return token.get(string2).compareTo(token.get(string1));
					}
			};

			TreeMap<String, Integer> freqSortToken = new TreeMap<String, Integer>(comparator);
			freqSortToken.putAll(token);
			return freqSortToken;
	}
	
	
//	------------------ Calculating Max Value ------------------
	
	public static ArrayList<Integer> calcMax(int input[])
	{
		int i;
		ArrayList<Integer> maxList = new ArrayList<Integer>();
		int max = 0;
		for(i = 0; i < input.length; i++)
		{
			if(input[i] > max)
			{
				maxList.clear();
				max = input[i];
				maxList.add(i);
			}
			else if(input[i] == max)
				maxList.add(i);
				
		}
		return maxList;
	}
	
	
//	------------------ Calculating terms with largest and lowest Document Frequency ------------------
	
	public static void calcLargestLowestTermDf (TreeMap<String, TermDocFreqNode> treeMap)
	{
		
		   ArrayList<String> largestTermDF = new ArrayList<String>();
		   ArrayList<String> lowestTermDF =  new ArrayList<String>();
		  int largestDf = -1;
		  int lowestDf = 999999999;
		  Iterator<String> iterator = treeMap.keySet().iterator();
		  		
		  		while(iterator.hasNext())
		  		{
		  			String s = iterator.next();
		  			TermDocFreqNode termDocFreqNode = treeMap.get(s);
		        
			        if(termDocFreqNode.dF > largestDf)
			        {
			        	largestTermDF.clear();
						largestDf=termDocFreqNode.dF;
					    largestTermDF.add(s);
					}
			        
			        else if(termDocFreqNode.dF==largestDf)
						largestTermDF.add(s);
					
					if(termDocFreqNode.dF<lowestDf)
					{
						lowestTermDF.clear();
						lowestDf=termDocFreqNode.dF;
						lowestTermDF.add(s);
					}
					
					else if(termDocFreqNode.dF == lowestDf)
						lowestTermDF.add(s);
					
		  		}
		  		System.out.println("	Max: "+largestTermDF.toString());
				System.out.println("	Min: "+lowestTermDF.toString());
	}
	
	
//	------------------ Compression of Posting List & Dictionary 1 ------------------
	
	public static ArrayList<TermDocFreqCompressedIndex> postingListandDictionaryCompressionOne(TreeMap<String, TermDocFreqNode> treeMap, int k) 
	{
		long compressionTime1 = new Date().getTime();
		int ptr = 0;
		int counter = -1;
		String s[] = new String[k];
		StringBuffer output = new StringBuffer();
		ArrayList<TermDocFreqCompressedIndex> compressedTermIndexList = new ArrayList<TermDocFreqCompressedIndex>();
		Iterator<String> iterator = treeMap.keySet().iterator();
		
		while (iterator.hasNext()) 
		{
			String word = iterator.next();
			TermDocFreqCompressedIndex termDocFreqCompressedIndex = postingListOneCompression(treeMap, word);
			if (counter == -1) 
			{
				termDocFreqCompressedIndex.ptr = ptr;
				counter = counter + 1;
				s[counter] = word;
				counter = counter + 1;
			} 
			
			else if (counter <= k - 1) 
			{
				s[counter] = word;
				counter = counter + 1;
			} 
			
			else 
			{
				String calculatedBlockCode = calcBlockCode(s);
				output.append(calculatedBlockCode);
				ptr = ptr + calculatedBlockCode.length();
				counter = -1;
			}
			
			compressedTermIndexList.add(termDocFreqCompressedIndex);
		}
		
		TermDocFreqCompressedIndex.frontCodeNode = output.toString();
		long compressionTime2 = new Date().getTime();
		compressedIndexTime1 = (compressionTime2 - compressionTime1);
		
		return compressedTermIndexList;
	}
	
	
//	------------------ Compression of Posting List & Dictionary 2 ------------------
	
	public static ArrayList<TermDocFreqCompressedIndex> postingListandDictionaryCompressionTwo (TreeMap<String, TermDocFreqNode> treeMap, int k) 
	{
		long compressionTime1 = new Date().getTime();
		int ptr = 0;
		int counter = -1;
		String s[] = new String[k];
		StringBuffer output = new StringBuffer();
		ArrayList<TermDocFreqCompressedIndex> compressedTermIndexList = new ArrayList<TermDocFreqCompressedIndex>();
		Iterator<String> iterator = treeMap.keySet().iterator();

		while (iterator.hasNext()) 
		{
			String word = iterator.next();
			TermDocFreqCompressedIndex termDocFreqCompressedIndex = postingListTwoCompression(treeMap, word);
			
			if (counter == -1) 
			{
				termDocFreqCompressedIndex.ptr = ptr;
				counter++;
				s[counter] = word;
				counter++;
			} 
			
			else if (counter <= k - 1) 
			{
				s[counter] = word;
				counter++;
			} 
			
			else 
			{
				String calculatedFrontCode = calcFrontCode(s);
				output.append(calculatedFrontCode);
				ptr = ptr + calculatedFrontCode.length();
				counter = -1;
			}
			
			compressedTermIndexList.add(termDocFreqCompressedIndex);
		}
		
		TermDocFreqCompressedIndex.frontCodeNode = output.toString();
		long compressionTime2 = new Date().getTime();
		compressedIndexTime2 = (compressionTime2 - compressionTime1);
		return compressedTermIndexList;
	}
	
	
//	------------------ Creating Index Function ------------------
	
	private static void creatingIndex(TreeMap<String, TermDocFreqNode> treeMap, String fPath) throws IOException
	{

		Iterator<String> iterator1 = treeMap.keySet().iterator();
		RandomAccessFile listIndx = new RandomAccessFile(fPath, "rw");
		listIndx.setLength(0);
		
		while(iterator1.hasNext())
		{
			String s = iterator1.next();
			TermDocFreqNode termDocFreqNode = treeMap.get(s);
			Iterator<Map.Entry<Integer, ArrayList<Integer>>> iterator2 = termDocFreqNode.postingList.entrySet().iterator();
			
			listIndx.writeBytes(s);
			listIndx.writeBytes("\t");
			listIndx.write(termDocFreqNode.dF);
			
			while (iterator2.hasNext()) 
			{
				Map.Entry<Integer, ArrayList<Integer>> mapEntry = iterator2.next();
				int documentID =  mapEntry.getKey();
				int tF = (mapEntry.getValue()).get(0);
				listIndx.write(documentID);
				listIndx.writeBytes("\t");
				listIndx.write(tF);
				listIndx.writeBytes("\t");
				listIndx.write((mapEntry.getValue()).get(1));
				listIndx.writeBytes("\t");
				listIndx.write((mapEntry.getValue()).get(2));
			}
			
			listIndx.writeBytes("\n");
		}
		
		listIndx.close();
	}
	
	
//	------------------ Creating Index Function ------------------
	
	private static void createIndex(ArrayList<TermDocFreqCompressedIndex> indexCompressedOne, String fPath) throws Exception 
	{
		int i, j;
		int length = indexCompressedOne.size();
		RandomAccessFile listIndx;
		listIndx = new RandomAccessFile(fPath, "rw");
		listIndx.setLength(0);
		listIndx.writeBytes(TermDocFreqCompressedIndex.frontCodeNode);
		
		for (i = 0; i < length; i++) 
		{
			TermDocFreqCompressedIndex termDocFreqCompressedIndex = indexCompressedOne.get(i);
			
			listIndx.write(termDocFreqCompressedIndex.dF);
			listIndx.write(termDocFreqCompressedIndex.ptr);
			
			for (j = 0; j < termDocFreqCompressedIndex.postingList.size(); j++) 
			{
				 listIndx.write(termDocFreqCompressedIndex.postingList.get(j).docID);
				 listIndx.write(termDocFreqCompressedIndex.postingList.get(j).tF);
				 listIndx.write(termDocFreqCompressedIndex.postingList.get(j).maxTF1);
				 listIndx.write(termDocFreqCompressedIndex.postingList.get(j).docLen);
			}
			
		}
		listIndx.close();
	}
	
	
//	------------------  ------------------
	
	public static void tokenization(File fileList[], ArrayList<String> stopWord, TreeMap<String, TermDocFreqNode> treeMap, TreeMap<String, TermDocFreqNode> treeMapStem) throws IOException
	{
		long uncompressedTime1, uncompressedTime2;
		uncompressedTime1 = new Date().getTime();
		int i, j, documentID=0;
		int length = fileList.length;

		for (i = 0; i < length; i++) 
		{
			if (fileList[i].isFile()) 
			{
				String name[]=fileList[i].getName().split("d");
				
				if(name[0]!=null && name[0].contains("cran"))
					documentID = Integer.parseInt(name[1]);
				
				String line = "";
				FileReader fileReader = new FileReader(fileList[i].getAbsoluteFile());
				@SuppressWarnings("resource")
				BufferedReader bufferedReader = new BufferedReader(fileReader);
				
				while ((line = bufferedReader.readLine()) != null) 
				{
					line = line.toLowerCase();
					line = line.replaceAll("\\<.*?>", "");
					line = line.replaceAll("[^A-Za-z\\s]", "").replaceAll("\\s+", " ");
					
					String documentText = line;
					Annotation document = new Annotation(documentText);

					pipeline1.annotate(document);

					List<CoreMap> sentences = document.get(SentencesAnnotation.class);
					String lemma = "";
					for (CoreMap sentence : sentences) 
					{
						for (CoreLabel token : sentence.get(TokensAnnotation.class)) 
						{
							lemma = (token.get(LemmaAnnotation.class));
							
							if (stopWord.contains(lemma)) 
							{
								continue;
							}
							
							String testLemma = lemma.trim();
							
							if (!testLemma.isEmpty()) 
							{
								if (treeMap.containsKey(testLemma)) 
								{
									TermDocFreqNode termDocFreqNode = treeMap.get(testLemma);
									termDocFreqNode.tF = termDocFreqNode.tF + 1;
									
									if (termDocFreqNode.postingList.containsKey(documentID)) 
									{
										Integer result = (termDocFreqNode.postingList.get(documentID)).get(0);
										result++;
										ArrayList<Integer> postingList = new ArrayList<Integer>();
										postingList.add(result);
										postingList.add(DocumentLength[documentID]);
										postingList.add(maxTermFreq[documentID]);
										termDocFreqNode.postingList.put(documentID, postingList);
									} 
									
									else 
									{
										ArrayList<Integer> postingList = new ArrayList<Integer>();
										int tF = 1;
										postingList.add(tF);
										postingList.add(DocumentLength[documentID]);
										postingList.add(maxTermFreq[documentID]);
										termDocFreqNode.postingList.put(documentID, postingList);
										termDocFreqNode.dF = termDocFreqNode.dF + 1;
									}
									
								} 
								
								else 
								{
									TermDocFreqNode termDocFreqNode = new TermDocFreqNode();
									termDocFreqNode.tF = 1;
									termDocFreqNode.postingList = new TreeMap<Integer, ArrayList<Integer>>();
									
									if (termDocFreqNode.postingList.containsKey(String.valueOf(documentID))) 
									{
										int result = (termDocFreqNode.postingList.get(documentID)).get(0);
										result++;

										ArrayList<Integer> postingList = new ArrayList<Integer>();
										postingList.add(result);
										postingList.add(DocumentLength[documentID]);
										postingList.add(maxTermFreq[documentID]);
										termDocFreqNode.postingList.put(documentID, postingList);
									} 
									
									else 
									{
										ArrayList<Integer> postingList = new ArrayList<Integer>();
										int tF = 1;
										postingList.add(tF);
										postingList.add(DocumentLength[documentID]);
										postingList.add(maxTermFreq[documentID]);
										termDocFreqNode.postingList.put(documentID, postingList);
										termDocFreqNode.dF = termDocFreqNode.dF + 1;
									}
									
									treeMap.put(testLemma, termDocFreqNode);
								}
							}
						}
					}

					String lemmas[] = line.split(" ");
					
					for (j=0; j<lemmas.length; j++) 
					{
						if (stopWord.contains(lemmas[j])) 
						{
							continue;
						}
						
						String lemmasTest = lemmas[j].trim();

						stemmer.add(lemmasTest.toCharArray(), lemmasTest.length());
						stemmer.stem();
						String wordStemmed = stemmer.toString();

						if (!wordStemmed.isEmpty()) 
						{
							if (treeMapStem.containsKey(wordStemmed)) 
							{
								TermDocFreqNode termDocFreqNode = treeMapStem.get(wordStemmed);
								termDocFreqNode.tF = termDocFreqNode.tF + 1;
								if (termDocFreqNode.postingList.containsKey(documentID)) 
								{
									int result = (termDocFreqNode.postingList.get(documentID)).get(0);
									result = result + 1;
									ArrayList<Integer> postingList = new ArrayList<Integer>();
									postingList.add(result);
									postingList.add(DocumentLength[documentID]);
									postingList.add(maxStemFreq[documentID]);
									termDocFreqNode.postingList.put(documentID, postingList);
								} 
								
								else 
								{
									ArrayList<Integer> postingList = new ArrayList<Integer>();
									int tF = 1;
									postingList.add(tF);
									postingList.add(DocumentLength[documentID]);
									postingList.add(maxStemFreq[documentID]);
									termDocFreqNode.postingList.put(documentID, postingList);
									termDocFreqNode.dF = termDocFreqNode.dF + 1;
								}
								
							}
							
							else 
							{
								TermDocFreqNode termDocFreqNode = new TermDocFreqNode();
								termDocFreqNode.tF = 1;
								termDocFreqNode.postingList = new TreeMap<Integer, ArrayList<Integer>>();
								
								if (termDocFreqNode.postingList.containsKey(documentID)) 
								{
									int result = (termDocFreqNode.postingList.get(documentID)).get(0);
									result = result + 1;

									ArrayList<Integer> postingList = new ArrayList<Integer>();
									postingList.add(result);
									postingList.add(DocumentLength[documentID]);
									postingList.add(maxStemFreq[documentID]);
									termDocFreqNode.postingList.put(documentID, postingList);

								} 
								
								else 
								{
									ArrayList<Integer> postingList = new ArrayList<Integer>();
									int tF = 1;
									postingList.add(tF);
									postingList.add(DocumentLength[documentID]);
									postingList.add(maxStemFreq[documentID]);
									termDocFreqNode.postingList.put(documentID, postingList);
									termDocFreqNode.dF += 1;
								}
								
								treeMapStem.put(wordStemmed, termDocFreqNode);
							}
						}
					}
				}
			}
		}

		uncompressedTime2 = new Date().getTime();
		uncompressedIndexTime = (uncompressedTime2 - uncompressedTime1) / 1000;
		return;
	}
	
	
	public static void main(String args[]) throws Exception
	{
		
		int i, j;
		String path = ".";
		String fPath = args[0].toString();
		String stopWordPath = args[1].toString();
		//String fPath = "/Users/karanmotani/Desktop/Cranfield";
		//String stopWordPath = "/Users/karanmotani/Desktop/stopwords.txt";
		
		TreeMap<String, TermDocFreqNode> treeMapStem = new TreeMap<String, TermDocFreqNode>();
		TreeMap<String, TermDocFreqNode> treeMap = new TreeMap<String, TermDocFreqNode>();
		File f1Uncompressed = new File(path + "/Uncompressed-Index1");
		File f1Compressed = new File(path + "/Compressed-Index1");
		File f2Uncompressed = new File(path + "/Uncompressed-Index2");
		File f2Compressed = new File(path + "/Compressed-Index2");
		
		File folder = new File(fPath);
		File[] listOfFiles = folder.listFiles();
		File stopWordFile = new File(stopWordPath);
		int length = listOfFiles.length;
		
		DocumentLength = new int [length+1];
		maxTermFreq = new int [length+1];
		maxStemFreq = new int [length+1];
		
		Arrays.fill(DocumentLength, 0);
		Arrays.fill(maxTermFreq, 0);
		Arrays.fill(maxStemFreq, 0);

		ArrayList<String> stopWords = stopWordsExtraction(stopWordFile);
		
		Properties property = new Properties();
		property.setProperty("annotators", "tokenize, ssplit, pos, lemma");
		pipeline1 = new StanfordCoreNLP(property);
		
		for (i = 0; i < length; i++) 
		{
			int documentLength1=0;
			int documentID=0;
			
			TreeMap<String, Integer> largestLemma = new TreeMap<String, Integer>();
			TreeMap<String, Integer> largestStem = new TreeMap<String, Integer>();
			
			if (listOfFiles[i].isFile()) 
			{
				String line = "";
				String[] fileName = listOfFiles[i].getName().split("d");
				
				if(fileName[0]!=null && fileName[0].contains("cran"))
					documentID=Integer.parseInt(fileName[1]);
				
				FileReader fr = new FileReader(listOfFiles[i].getAbsoluteFile());
				@SuppressWarnings("resource")
				BufferedReader br = new BufferedReader(fr);
				while ((line = br.readLine()) != null) 
				{
					line = line.toLowerCase();
					line = line.replaceAll("\\<.*?>", "");
					line = line.replaceAll("[^A-Za-z\\s]", "").replaceAll("\\s+", " ");
					
					String documentText = line;
					
					Annotation document = new Annotation(documentText);
					pipeline1.annotate(document);
					List<CoreMap> sentences = document.get(SentencesAnnotation.class);
					
					String lemma = "";
					for (CoreMap sentence : sentences) 
					{
						for (CoreLabel token : sentence.get(TokensAnnotation.class)) 
						{
							lemma = token.get(LemmaAnnotation.class);
							documentLength1++;
							if (!stopWords.contains(lemma) && !lemma.trim().isEmpty()) 
							{
								if (largestLemma.containsKey(lemma))
									largestLemma.put(lemma, largestLemma.get(lemma) + 1);
								else
									largestLemma.put(lemma, 1);
							}
						}
		
						String[] lemmas = line.split(" ");
						Stemmer stemmer= new Stemmer();
						
						for (j = 0; j < lemmas.length; j++) 
						{
							if (!stopWords.contains(lemmas[j])) {
							String stemChars = lemmas[j].trim();
							stemmer.add(stemChars.toCharArray(), stemChars.length());
							stemmer.stem();
						if(largestStem.get(stemmer.toString()) == null)
								largestStem.put(stemmer.toString(), 1);
						else
								largestStem.put(stemmer.toString(), largestStem.get(stemmer.toString()) + 1);
							}
						}
					}
				}
			
				largestLemma = freqSort(largestLemma);
				largestStem = freqSort(largestStem);
				maxStemFreq[documentID]= largestStem.firstEntry().getValue();
				maxTermFreq[documentID]= largestLemma.firstEntry().getValue();
				DocumentLength[documentID]= documentLength1;
				
			}
		}
		
		tokenization(listOfFiles, stopWords, treeMap, treeMapStem);
		
		ArrayList<TermDocFreqCompressedIndex> indexCompressedOne = postingListandDictionaryCompressionOne(treeMap, 8);
		ArrayList<TermDocFreqCompressedIndex> indexCompressedTwo = postingListandDictionaryCompressionTwo(treeMapStem, 8);
		
		creatingIndex(treeMap, f1Uncompressed.getAbsolutePath());
		creatingIndex(treeMapStem, f2Uncompressed.getAbsolutePath());
		createIndex(indexCompressedOne, f1Compressed.getAbsolutePath());
		createIndex(indexCompressedTwo, f2Compressed.getAbsolutePath());
		
		System.out.println("1.  Elapsed time required to build any Uncompressed version of the index: " + uncompressedIndexTime+"s");
		System.out.println("	Elapsed time required to build Compressed version 1 of the index: " + compressedIndexTime1+"ms");
		System.out.println("	Elapsed time required to build Compressed version 2 of the index: " + compressedIndexTime2+"ms");
		
		System.out.println("2.  The size of the index Version 1 uncompressed (in bytes): " + f1Uncompressed.length());
		
		System.out.println("3.  The size of the index Version 2 uncompressed (in bytes): " + f2Uncompressed.length());
		
		System.out.println("4.  The size of the index Version 1 compressed (in bytes): " + f1Compressed.length());
		
		System.out.println("5.  The size of the index Version 2 compressed (in bytes): " + f2Compressed.length());
		
		System.out.println("6. 	The number of inverted lists in version 1 uncompressed: " + treeMap.size());
		System.out.println("	The number of inverted lists in version 2 uncompressed: " + treeMapStem.size());
		System.out.println("	The number of inverted lists in version 1 compressed: " + indexCompressedOne.size());
		System.out.println("	The number of inverted lists in version 2 compressed: " + indexCompressedTwo.size());
		
		System.out.println("7.");
		String words[] = {"Reynolds", "NASA", "Prandtl", "flow", "pressure", "boundary", "shock"};
		for (i = 0; i < words.length; i++) 
		{
			String wordTest = words[i].toLowerCase();
			stemmer.add(wordTest.toCharArray(), wordTest.length());
			stemmer.stem();
			String stemmedWord = stemmer.toString();
			TermDocFreqNode node = treeMapStem.get(stemmedWord);
			System.out.println("	Term: " + wordTest);
			System.out.println("	Document Frequency: " + node.dF);
			System.out.println("	Term Frequency: " + node.tF);
			System.out.println("	Inverted List length (in bytes): " + node.postingList.size() * 2 * Integer.BYTES);
			System.out.println();
		}
		
		TermDocFreqNode termDocFreqNode = treeMap.get("nasa");
		System.out.println("8.	nasa");
		System.out.println("	Term freq of 'nasa': " + termDocFreqNode.tF);
		System.out.println("	Doc Freq of 'nasa': " + termDocFreqNode.dF);
		
		Iterator<Map.Entry<Integer, ArrayList<Integer>>> iterator = termDocFreqNode.postingList.entrySet().iterator();
		
		
		for (i=0;i<3;i++) 
		{
			  iterator.hasNext();
			  Map.Entry<Integer, ArrayList<Integer>> mapEntry = iterator.next();
			  int documentID =  mapEntry.getKey();
			  int tF = (mapEntry.getValue()).get(0);
			  int docLen = (mapEntry.getValue()).get(1);
			  int largest_tf = (mapEntry.getValue()).get(2);
			  System.out.println("	Document ID: "+documentID +"  term_freq: " +tF + "  Max_tf: "+ largest_tf +"  doc_Len: "+docLen);
			}
		
		System.out.println("\n9.	Lemma with the Minimum and Maximum Document Freq: ");
			calcLargestLowestTermDf(treeMap);
		System.out.println("\n10.	Stem with the Minimum and Maximum Document Freq::");
			calcLargestLowestTermDf(treeMapStem);
		
		ArrayList<Integer> DocID_MaxLen = calcMax(DocumentLength);
		ArrayList<Integer> DocID_maxTF = calcMax(maxTermFreq);
		ArrayList<Integer> DocID_maxSF = calcMax(maxStemFreq);
		
		System.out.println("11. The document with the largest max_tf in collection(lemma): "+DocID_maxTF.toString());
		System.out.println("	The document with the largest max_tf in collection(stem): "+DocID_maxSF.toString());
		System.out.println("	The document with the largest doclen in the collection: "+DocID_MaxLen.toString());

		
	}
	
	
}