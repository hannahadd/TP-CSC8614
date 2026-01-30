# Re: Demande de relecture

**From:** Eric Noulard <enoulard@antidot.net>

**Date:** Fri, 16 Jan 2026 14:50:25 +0100

**Message-ID:** <47c6edd3c854e985edcb3a85d6de7230@antidot.net>

---

Le 12.01.2026 11:40, Tatiana NIAURONIS a écrit :
> Bonjour Monsieur,
> 
> Nous espérons que vous allez bien.
> 
> Comme convenu, nous avons travaillé de notre côté le document
> "Méthodologie fusionnée"

J'ai relu le doc fusionné et fais quelques remarques dedans.
Ce document est plutôt bien mais vous devriez améliorer et vérifier la 
manière dont vous citer la biblio
car les références bibliographique ne sont pas systématiquement citées 
avec leur référence numérotée i.e. [3]
qui est la bonne manière de garantir qu'on peut vérifier quelle est la 
source de ce qui est dit.

D'autre part la biblio nécessite un peu de mise en forme:

- supprimer les doublons
- faite apparaitre le contenu des liens Web de manière à ce qu'une 
version non clickable ou imprimée permette de lire les liens

Une dernière chose, vous êtes les auteurs de ce rapport dont la liste 
des auteurs doit apparaitre
par exemple sur une page seule après la page de titre.

> ainsi que le repository sur Github qui est
> disponible à ce lien: https://github.com/LaurealDente/env5001.

Concernant votre code python, il pourrait avoir la structure standard 
d'un module python autonome.
Vous pourriez indiquer plus explicitement les dépendances (uvicorn, 
fastapi, Pyyaml)grace à ça.
Je vous conseille d'utiliser des moyens standard python pour ça, i.E. 
ecrire un pyproject.toml
utiliser un outils de gestion de dependence moderne comme uv: 
https://docs.astral.sh/uv/guides/projects/

Sinon l'idée d'écrire une petite API REST pour faire ces calculs est 
intéressante/bonne.
Votre endpoint /energy devrait être un POST et pas un GET dans lequel on 
passe le fichier de donner dans le body du POST, i.e. votre appli ne 
contient pas les données analytics de Fluid Topics on les lui passe dans 
la requête HTTP.


> Nous avons également créé un document Questions réponses afin de
> faciliter le suivi de vos réponses.

ok vu.

> 
> Vous remerciant par avance,
> 
> Cordialement,
> 
> Le groupe :)
> 
> 
> 
> ----- Mail original -----
> De: "Eric Noulard" <enoulard@antidot.net>
> À: "NIAURONIS Tatiana" <tatiana.niauronis@telecom-sudparis.eu>
> Cc: "Corentin MONGIN" <corentin.mongin@telecom-sudparis.eu>, "HADDAOUI
> Hanna" <hanna.haddaoui@telecom-sudparis.eu>, "JERBI Salim"
> <salim.jerbi@telecom-sudparis.eu>, "Yohan DELIERE"
> <yohan.deliere@telecom-sudparis.eu>, "LAURET Alexandre"
> <alexandre.lauret@telecom-sudparis.eu>, "alexandre galstian"
> <alexandre.galstian@telecom-sudparis.eu>, "SLESINSKI Robin"
> <robin_slesinski@telecom-sudparis.eu>
> Envoyé: Vendredi 2 Janvier 2026 16:31:49
> Objet: Re: Lien visio
> 
> Re-Bonjour,
> 
> Avez les dates et heures de votre soutenance finale?
> Je ne pourrais pas être là en présentiel mais j'aimerais réserver le
> créneau pour être en mesure d'être là en visio.
> 
> Une fois que vous avez ces infos merci de me les communiquer 
> rapidement,
> Eric

-- 
Eric Noulard
Research Team
enoulard@antidot.net
Tél : +33 4 72 76 31 49
www.antidot.net | twitter.com/AntidotNet
