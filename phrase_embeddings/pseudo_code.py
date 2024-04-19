value = df.at[0, 'responsibilities']
print(value)

value_converted = []
for i in value:
  value_converted.append(i.split(' '))
print(value_converted)

''' the following is pseudo code, bc we are hoping to be able to use a pre-trained model as a word registry, 
but if we are not then we are going to need code where we operate on each element in order to generate our registry,
which is not ideal'''
value_converted_numeric = []
for i in value_converted:
  phrase_numeric_list = []
  for j in i:
    if j in registry:
      embdg = registry[j]
      phrase_numeric_list.append(embdg)
    else:
      j = j.replace('/', ' ')
      j = j.replace('-', ' ')
      j = j.replace(',', '')
      j_split = j.split(' ')
      if len(j_split) > 1:
        for i in j_split:
          if i in registry:
            embdg = registry[i]
            phrase_nuberic_list.append(embdg)
          else:
            pass
      else:
        pass


  value_converted_numeric.append(phrase_numeric_list)